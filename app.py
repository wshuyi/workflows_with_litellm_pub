import argparse
from pathlib import Path
import sys
import pyperclip
from dotenv import load_dotenv
import os
import yaml
import json
from typing import List, Dict, Any, Union, Callable
import re
import time
from tqdm import tqdm
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
from exa_py import Exa
from litellm import completion

# Utils
def setup_logging(log_dir='logs'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'text_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,  # 将级别改为 INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

@contextmanager
def open_file(file_path: str, mode: str):
    try:
        file = open(file_path, mode, encoding='utf-8')
        yield file
    finally:
        file.close()

logger = setup_logging()

# API Client
class APIClient:
    def __init__(self, model: str):
        load_dotenv(override=True)
        self.model = model
        self.api_base = None
        self.api_key = None

        if model.startswith("openai/"):
            self.api_base = os.getenv('OPENAI_API_BASE')
            self.api_key = os.getenv('OPENAI_API_KEY')

    def query_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            kwargs = {}
            if self.api_base:
                kwargs['api_base'] = self.api_base
            if self.api_key:
                kwargs['api_key'] = self.api_key

            response = completion(
                model=self.model,
                messages=messages,
                # temperature=0.1,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling API for model {self.model}: {e}")
            raise

# Config Manager
class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        required_keys = ['strategies']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
        
        if not isinstance(config['strategies'], list):
            raise ValueError("'strategies' must be a list")
        
        for strategy in config['strategies']:
            if 'tool_name' not in strategy and 'model' not in strategy:
                raise ValueError("Each strategy must have either 'tool_name' or 'model'")

class ConfigManager:
    @staticmethod
    def load_config(workflow_name: str, custom_config_path: str = None) -> Dict[str, Any]:
        if custom_config_path:
            config_path = custom_config_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config', f'{workflow_name}_config.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            ConfigValidator.validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

# Processing Strategy
class StrategyConfig:
    def __init__(self, tool_name: str = None, model: str = None, prompt_name: str = None, 
                 input_format: str = None, output_name: str = None, tool_params: Dict[str, Any] = None):
        self.tool_name = tool_name
        self.model = model
        self.prompt_name = prompt_name
        self.input_format = input_format
        self.output_name = output_name
        self.tool_params = tool_params or {}

class ProcessingStrategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = StrategyConfig(**config)
        self.tool_params = self.config.tool_params

    def process(self, chunk: str, processor: 'TextProcessor', previous_outputs: Dict[str, str]) -> str:
        logger.debug(f"Processing strategy with config: {self.config.__dict__}")
        
        try:
            if self.config.tool_name:
                result = processor.execute_tool(self.config.tool_name, chunk, self.tool_params)
            elif self.config.model:
                prompt = self.prepare_prompt(chunk, processor, previous_outputs)
                result = processor.execute_model(self.config.model, prompt)
            else:
                raise ValueError("Neither tool_name nor model is specified in the strategy configuration")
            
            if result is None:
                logger.warning(f"Strategy {self.config.prompt_name or self.config.tool_name} returned None")
                return ""
            return result
        except Exception as e:
            logger.error(f"Error in strategy {self.config.prompt_name or self.config.tool_name}: {str(e)}")
            return ""

    def prepare_prompt(self, chunk: str, processor: 'TextProcessor', previous_outputs: Dict[str, str]) -> str:
        prompt_template = processor.read_system_prompt(self.config.prompt_name)
        input_format = self.config.input_format

        for key, value in previous_outputs.items():
            input_format = input_format.replace(f"{{{{{key}}}}}", f"{{{{{value}}}}}")
        
        input_format = input_format.replace("{{text}}", f"{{{{{chunk}}}}}")
        input_format = input_format.replace("{{memory_vocab}}", f"{{{{{processor.memory.get('vocab', '')}}}}}")
        input_format = input_format.replace("{{memory_blog_example}}", f"{{{{{processor.memory.get('blog_example', '')}}}}}")

        return f"{prompt_template}\n\n{input_format}"

# Text Processor
class BaseTextProcessor:
    def __init__(self, max_tokens_per_chunk: int = 1000):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        self.config = {}
        self.chunk_count = 0
        self.current_chunk_number = 0
        self.memory = {}
        self.load_memory_files()

    def preprocess_text(self, text: str) -> str:
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            processed_lines = []
            for i, line in enumerate(lines):
                if i == 0 or not line.strip():
                    processed_lines.append(line)
                elif (len(line) > 0 and not line[0].isupper() and not line[0].isdigit() and
                      i > 0 and len(lines[i-1].strip()) > 0 and
                      lines[i-1].strip()[-1] not in '.!?.!?]'):
                    processed_lines[-1] += ' ' + line.strip()
                else:
                    processed_lines.append(line)
            processed_paragraphs.append('\n'.join(processed_lines))
        return '\n\n'.join(processed_paragraphs)

    def split_text(self, text: str) -> List[str]:
        preprocessed_text = self.preprocess_text(text)
        chars_per_token = len(preprocessed_text) / len(self.encoding.encode(preprocessed_text))
        max_chars = int(self.max_tokens_per_chunk * chars_per_token)
        
        logger.info(f"Preprocessed text length: {len(preprocessed_text)}")
        logger.info(f"Chars per token: {chars_per_token:.2f}")
        logger.info(f"Max chars per chunk: {max_chars}")
        logger.info(f"Max tokens per chunk: {self.max_tokens_per_chunk}")
        
        separators = ["\n\n", "\n", ". ", "!", "?", ";", ",", ".", "!", "?", ";", ",", " ", ""]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=20,
            length_function=lambda t: len(self.encoding.encode(t)),
            separators=separators
        )
        
        chunks = text_splitter.split_text(preprocessed_text)
        
        logger.info(f"Number of chunks after initial split: {len(chunks)}")
        
        result = [self._split_chunk(chunk) if len(self.encoding.encode(chunk)) > self.max_tokens_per_chunk else chunk for chunk in chunks]
        
        logger.info(f"Final number of chunks: {len(result)}")
        
        return result

    def _split_chunk(self, chunk: str) -> str:
        sentences = re.split(r'(?<=[. !?.!?])\s*', chunk)
        sub_chunks = []
        current_sub_chunk = []
        current_token_count = 0
        for sentence in sentences:
            sentence_token_count = len(self.encoding.encode(sentence))
            if sentence_token_count > self.max_tokens_per_chunk:
                words = sentence.split()
                for word in words:
                    word_token_count = len(self.encoding.encode(word))
                    if current_token_count + word_token_count > self.max_tokens_per_chunk:
                        if current_sub_chunk:
                            sub_chunks.append(" ".join(current_sub_chunk))
                        current_sub_chunk = [word]
                        current_token_count = word_token_count
                    else:
                        current_sub_chunk.append(word)
                        current_token_count += word_token_count
            elif current_token_count + sentence_token_count > self.max_tokens_per_chunk:
                if current_sub_chunk:
                    sub_chunks.append(" ".join(current_sub_chunk))
                current_sub_chunk = [sentence]
                current_token_count = sentence_token_count
            else:
                current_sub_chunk.append(sentence)
                current_token_count += sentence_token_count
        if current_sub_chunk:
            sub_chunks.append(" ".join(current_sub_chunk))
        return " ".join(sub_chunks)

    def read_system_prompt(self, pattern_name: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(current_dir, 'patterns', pattern_name, 'system.md')
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except IOError as e:
            logger.error(f"Error reading system prompt: {e}")
            raise

    def load_memory_files(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        memory_dir = os.path.join(current_dir, 'memory')
        if not os.path.exists(memory_dir):
            logger.warning(f"Memory directory not found: {memory_dir}")
            return

        for file_name in os.listdir(memory_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(memory_dir, file_name)
                key = file_name[:-3]
                try:
                    with open_file(file_path, 'r') as f:
                        self.memory[key] = f.read().strip()
                    logger.info(f"Loaded memory file: {file_name}")
                except IOError as e:
                    logger.error(f"Error reading memory file {file_name}: {e}")

# Tools
class SearchTools:
    @staticmethod
    def exa_search(query: str, category: str = "tweet", start_date: str = None) -> str:
        logger.info(f"Executing Exa search with query: {query}, category: {category}")
        exa = Exa(os.getenv("EXA_API_KEY"))
        
        if start_date is None and category == "tweet":
            start_date = (datetime.now() - timedelta(hours=72)).strftime("%Y-%m-%d")
        
        try:
            results = exa.search_and_contents(
                query,
                num_results=10,
                start_published_date=start_date,
                use_autoprompt=True,
                category=category,
                text={"max_characters": 2000},
                highlights={"highlights_per_url": 2, "num_sentences": 1, "query": f"This is the highlight query: {query}"}
            )
            
            logger.info(f"Exa {category} search completed successfully")
            return f'# Topic: {query}\n{str(results)}'
        except Exception as e:
            logger.error(f"Error in Exa {category} search: {str(e)}")
            return f"Error in Exa {category} search: {str(e)}"

class TextProcessor(BaseTextProcessor):
    def __init__(self, config: Dict[str, Any], max_tokens_per_chunk: int = 1000, verbose: bool = False, debug: bool = False):
        super().__init__(max_tokens_per_chunk)
        self.config = config
        self.verbose = verbose
        self.debug = debug
        self.tools = self.load_tools()
        self.models = self.load_models()
        self.strategies = self.load_strategies()
        
        if self.debug:
            logger.info("Initialized TextProcessor")

    def load_tools(self) -> Dict[str, Callable]:
        tools = {}
        for strategy in self.config.get('strategies', []):
            if 'tool_name' in strategy:
                tool_name = strategy['tool_name']
                if tool_name not in tools:
                    if tool_name == 'exa_search':
                        tools[tool_name] = SearchTools.exa_search
                    elif hasattr(self, tool_name):
                        tools[tool_name] = getattr(self, tool_name)
                    else:
                        # 尝试从全局命名空间导入
                        tool = globals().get(tool_name)
                        if tool:
                            tools[tool_name] = tool
                        else:
                            raise ValueError(f"Tool '{tool_name}' not found")
        logger.debug(f"Loaded tools: {tools}")
        return tools

    def load_models(self) -> Dict[str, Any]:
        models = {}
        for strategy in self.config.get('strategies', []):
            if 'model' in strategy:
                model_name = strategy['model']
                if model_name not in models:
                    models[model_name] = APIClient(model_name)
        return models

    def load_strategies(self) -> List[ProcessingStrategy]:
        return [ProcessingStrategy(strategy_config) for strategy_config in self.config.get('strategies', [])]

    def process_chunk(self, chunk: str) -> str:
        self.current_chunk_number += 1
        if self.debug:
            logger.info(f"Processing chunk {self.current_chunk_number}/{self.chunk_count}")
        previous_outputs = {}
        final_result = ""
        for i, strategy in enumerate(self.strategies, 1):
            if self.debug:
                logger.info(f"Applying strategy {i}: {strategy.config.prompt_name}")
            try:
                result = strategy.process(chunk, self, previous_outputs)
                if result is None:
                    if self.debug:
                        logger.warning(f"Strategy {i} returned None")
                    continue
                previous_outputs[strategy.config.output_name] = result
                final_result = result
                if self.debug:
                    logger.info(f"Result of strategy {i}: {result[:100]}...")  # 只显示结果的前100个字符
            except Exception as e:
                logger.error(f"Error in strategy {i}: {str(e)}")
        return final_result

    def process_text(self, text: str) -> List[str]:
        chunks = self.split_text(text)
        self.chunk_count = len(chunks)
        logger.info(f"Text split into {self.chunk_count} chunks.")
        return [self.process_chunk(chunk) for chunk in tqdm(chunks, desc="Processing chunks")]

    def execute_model(self, model_name: str, input_text: str) -> str:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        client = self.models[model_name]
        
        current_strategy = next(s for s in self.strategies if s.config.model == model_name)
        prompt_name = current_strategy.config.prompt_name
        
        system_message = self.read_system_prompt(prompt_name)
        user_message = input_text
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        if self.debug:
            logger.info(f"Executing model: {model_name}")
        
        return client.query_api(messages)

    def execute_tool(self, tool_name: str, chunk: str, tool_params: Dict[str, Any] = None) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        if tool_params:
            return tool(chunk, **tool_params)
        return tool(chunk)

class ProcessorFactory:
    @staticmethod
    def create_processor(config: Dict[str, Any], max_tokens: int, verbose: bool, debug: bool) -> TextProcessor:
        return TextProcessor(config, max_tokens, verbose, debug)

# Main application
def save_output(results, output_path, output_format):
    output_handlers = {
        "json": lambda data, file: json.dump(data, file, indent=2),
        "md": lambda data, file: file.write("\n\n".join(filter(None, data))),
        "txt": lambda data, file: file.write("\n\n".join(filter(None, data)))
    }
    
    with open_file(output_path, "w") as f:
        output_handlers[output_format](results, f)

def main():
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Process text with configurable workflows.")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("--workflow", type=str, required=True, help="Workflow type")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Maximum tokens per chunk")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console output")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--output_format", type=str, default="md", choices=["md", "txt", "json"], help="Output file format")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed prompt logging")
    args = parser.parse_args()

    try:
        config = ConfigManager.load_config(args.workflow, args.config)
        if args.debug:
            logger.info("Config loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)
    
    processor = ProcessorFactory.create_processor(config, args.max_tokens, args.verbose, args.debug)

    input_path = Path(args.input_file).resolve()
    output_path = input_path.parent / f"{args.workflow}-output.{args.output_format}"

    try:
        with open_file(input_path, "r") as f:
            text = f.read()

        if args.debug:
            logger.info("Input text loaded successfully")
        results = processor.process_text(text)

        if not results:
            logger.warning("No results generated")
        else:
            save_output(results, output_path, args.output_format)
            if args.debug:
                logger.info(f"Results saved to {output_path}")

            with open_file(output_path, "r") as f:
                content = f.read()
            pyperclip.copy(content)
            if args.debug:
                logger.info("Content copied to clipboard")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Invalid input: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()