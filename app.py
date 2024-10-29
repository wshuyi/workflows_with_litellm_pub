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

# Utility Functions
@contextmanager
def open_file_utf8(file_path: str, mode: str):
    """Safe file handling with UTF-8 encoding"""
    try:
        file = open(file_path, mode, encoding='utf-8')
        yield file
    finally:
        file.close()

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Set up logging with file and console handlers"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'text_processor_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration Classes
class StrategyConfig:
    """Configuration for a single processing strategy"""
    def __init__(self, tool_name: str = None, model: str = None,
                 prompt_name: str = None, input_format: str = None,
                 output_name: str = None, tool_params: Dict[str, Any] = None):
        self.tool_name = tool_name
        self.model = model
        self.prompt_name = prompt_name
        self.input_format = input_format
        self.output_name = output_name
        self.tool_params = tool_params or {}

class ConfigValidator:
    """Validates configuration structure and content"""
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
        
        if 'parameters' in config:
            if not isinstance(config['parameters'], dict):
                raise ValueError("'parameters' must be a dictionary")
            if 'tokens' in config['parameters'] and not isinstance(config['parameters']['tokens'], int):
                raise ValueError("'tokens' in parameters must be an integer")

class ConfigManager:
    """Manages configuration loading and validation"""
    @staticmethod
    def load_config(workflow_name: str, custom_config_path: str = None) -> Dict[str, Any]:
        if custom_config_path:
            config_path = custom_config_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config', f"{workflow_name}.yaml")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found for workflow: {workflow_name}")
        
        try:
            with open_file_utf8(config_path, 'r') as f:
                config = yaml.safe_load(f)
            ConfigValidator.validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

# API Client
class APIClient:
    """Client for API interactions"""
    def __init__(self, model: str):
        load_dotenv(override=True)
        self.model = model
        self.api_base = None
        self.api_key = None

        if model.startswith("openai/"):
            self.api_base = os.getenv('OPENAI_API_BASE')
            self.api_key = os.getenv('OPENAI_API_KEY')

    def query_api(self, messages: List[Dict[str, str]]) -> str:
        """Send query to API and handle response"""
        try:
            kwargs = {}
            if self.api_base:
                kwargs['api_base'] = self.api_base
            if self.api_key:
                kwargs['api_key'] = self.api_key

            response = completion(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling API for model {self.model}: {e}")
            raise

# Text Processing
class BaseTextProcessor:
    """Base class for text processing functionality"""
    def __init__(self, config: Dict[str, Any], default_max_tokens: int = 1000):
        self.max_tokens_per_chunk = config.get('parameters', {}).get('tokens', default_max_tokens)
        self.encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        self.config = config
        self.chunk_count = 0
        self.current_chunk_number = 0
        self.memory = {}
        self.load_memory_files()

    def preprocess_text(self, text: str) -> str:
        """Preprocess text before splitting into chunks"""
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            processed_lines = []
            for i, line in enumerate(lines):
                if i == 0 or not line.strip():
                    processed_lines.append(line)
                elif (len(line) > 0 and not line[0].isupper() and 
                      not line[0].isdigit() and i > 0 and 
                      len(lines[i-1].strip()) > 0 and 
                      lines[i-1].strip()[-1] not in '.!?.!?]'):
                    processed_lines[-1] += ' ' + line.strip()
                else:
                    processed_lines.append(line)
            processed_paragraphs.append('\n'.join(processed_lines))
        return '\n\n'.join(processed_paragraphs)

    def split_text(self, text: str) -> List[str]:
        """Split text into processable chunks"""
        preprocessed_text = self.preprocess_text(text)
        chars_per_token = len(preprocessed_text) / len(self.encoding.encode(preprocessed_text))
        max_chars = int(self.max_tokens_per_chunk * chars_per_token)
        
        logger.info(f"Preprocessed text length: {len(preprocessed_text)}")
        logger.info(f"Chars per token: {chars_per_token:.2f}")
        logger.info(f"Max chars per chunk: {max_chars}")
        
        separators = ["\n\n", "\n", ". ", "!", "?", ";", ",", ".", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=20,
            length_function=lambda t: len(self.encoding.encode(t)),
            separators=separators
        )
        
        chunks = text_splitter.split_text(preprocessed_text)
        logger.info(f"Number of chunks after initial split: {len(chunks)}")
        
        return [self._split_chunk(chunk) if len(self.encoding.encode(chunk)) > 
                self.max_tokens_per_chunk else chunk for chunk in chunks]

    def _split_chunk(self, chunk: str) -> str:
        """Split a chunk if it exceeds token limit"""
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        current_chunk = []
        current_tokens = 0
        chunks = []

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            if sentence_tokens > self.max_tokens_per_chunk:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                chunks.append(sentence[:self.max_tokens_per_chunk])
                current_chunk = []
                current_tokens = 0
            elif current_tokens + sentence_tokens > self.max_tokens_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return " ".join(chunks)

    def load_memory_files(self):
        """Load memory files from the memory directory"""
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
                    with open_file_utf8(file_path, 'r') as f:
                        self.memory[key] = f.read().strip()
                    logger.info(f"Loaded memory file: {file_name}")
                except Exception as e:
                    logger.error(f"Error reading memory file {file_name}: {e}")

class TextProcessor(BaseTextProcessor):
    """Main text processing class"""
    def __init__(self, config: Dict[str, Any], default_max_tokens: int = 1000,
                 verbose: bool = False, debug: bool = False):
        super().__init__(config, default_max_tokens)
        self.verbose = verbose
        self.debug = debug
        self.tools = self.load_tools()
        self.models = self.load_models()
        self.strategies = self.load_strategies()
        self.current_strategy = None
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_dir = self._create_run_log_dir()
        self.current_chunk_step = 0  # 添加当前chunk的step计数器

    def _create_run_log_dir(self) -> str:
        """创建运行日志目录"""
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs', 
            'runs',
            self.run_timestamp
        )
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def log_strategy_step(self, strategy_name: str, step_number: int, 
                         input_data: Any, output_data: Any, error: str = None,
                         model_name: str = None):
        """记录策略步骤的输入输出"""
        self.current_chunk_step += 1  # 增加当前chunk的step计数
        
        # 构建文件名前缀，使用current_chunk_step作为步骤编号
        file_prefix = f'chunk_{self.current_chunk_number:02d}_step_{self.current_chunk_step:02d}_{strategy_name}'
        if model_name:
            file_prefix = f'{file_prefix}_{model_name.replace("/", "_")}'
        
        # 记录输入
        input_path = os.path.join(self.run_log_dir, f'{file_prefix}_input.md')
        with open_file_utf8(input_path, 'w') as f:
            if isinstance(input_data, (dict, list)):
                f.write("```json\n")
                f.write(json.dumps(input_data, indent=2, ensure_ascii=False))
                f.write("\n```")
            else:
                f.write(str(input_data))

        # 记录输出
        output_path = os.path.join(self.run_log_dir, f'{file_prefix}_output.md')
        with open_file_utf8(output_path, 'w') as f:
            if isinstance(output_data, (dict, list)):
                f.write("```json\n")
                f.write(json.dumps(output_data, indent=2, ensure_ascii=False))
                f.write("\n```")
            else:
                f.write(str(output_data))

        # 如果有错误，记录错误
        if error:
            error_path = os.path.join(self.run_log_dir, f'{file_prefix}_error.md')
            with open_file_utf8(error_path, 'w') as f:
                f.write(f"# Error\n\n{error}")

        logger.info(f"Chunk {self.current_chunk_number}, Step {self.current_chunk_step} ({strategy_name}) logged to: {self.run_log_dir}")

    def load_tools(self) -> Dict[str, Callable]:
        """Load available tools"""
        tools = {}
        for strategy in self.config.get('strategies', []):
            if 'tool_name' in strategy:
                tool_name = strategy['tool_name']
                if tool_name == 'exa_search':
                    tools[tool_name] = SearchTools.exa_search
        return tools

    def load_models(self) -> Dict[str, APIClient]:
        """Load available models"""
        models = {}
        for strategy in self.config.get('strategies', []):
            if 'model' in strategy:
                model_name = strategy['model']
                if model_name not in models:
                    models[model_name] = APIClient(model_name)
        return models

    def load_strategies(self) -> List[Any]:
        """Load processing strategies"""
        return [ProcessingStrategy(strategy_config) 
                for strategy_config in self.config.get('strategies', [])]

    def process_text(self, text: str) -> List[str]:
        """Process entire text through the pipeline"""
        chunks = self.split_text(text)
        self.chunk_count = len(chunks)
        logger.info(f"Text split into {self.chunk_count} chunks")
        return [self.process_chunk(chunk) for chunk in tqdm(chunks, desc="Processing chunks")]

    def process_chunk(self, chunk: str) -> str:
        """Process a single chunk through all strategies"""
        self.current_chunk_number += 1
        self.current_chunk_step = 0  # 重置当前chunk的step计数器
        if self.debug:
            logger.info(f"Processing chunk {self.current_chunk_number}/{self.chunk_count}")
        
        previous_outputs = {}
        final_result = ""
        
        for i, strategy in enumerate(self.strategies, 1):
            self.current_strategy = strategy
            try:
                result = strategy.process(chunk, self, previous_outputs)
                if result is not None:
                    previous_outputs[strategy.config.output_name] = result
                    final_result = result
                    if self.debug:
                        logger.info(f"Strategy {i} result: {result[:500]}...")
            except Exception as e:
                logger.error(f"Error in strategy {i}: {str(e)}")
        
        return final_result

    def execute_model(self, model_name: str, input_text: str) -> str:
        """Execute a model-based strategy"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        client = self.models[model_name]
        
        # 准备消息
        if isinstance(input_text, str):
            system_message = self.read_system_prompt(self.current_strategy.config.prompt_name)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text}
            ]
        else:
            messages = input_text

        try:
            # 执行 API 调用
            response = client.query_api(messages)
            
            # 记录这一步的输入输出
            step_number = len(os.listdir(self.run_log_dir)) + 1
            strategy_name = self.current_strategy.config.prompt_name or "model_call"
            
            self.log_strategy_step(
                strategy_name=strategy_name,
                step_number=step_number,
                input_data=messages,
                output_data={"response": response},
                model_name=model_name
            )
            
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model execution error: {error_msg}")
            
            # 记录错误
            step_number = len(os.listdir(self.run_log_dir)) + 1
            strategy_name = self.current_strategy.config.prompt_name or "model_call"
            
            self.log_strategy_step(
                strategy_name=strategy_name,
                step_number=step_number,
                input_data=messages,
                output_data=None,
                error=error_msg,
                model_name=model_name
            )
            
            raise

    def execute_tool(self, tool_name: str, chunk: str, tool_params: Dict[str, Any] = None) -> str:
        """Execute a tool-based strategy"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        merged_params = {**self.current_strategy.tool_params, **(tool_params or {})}
        
        try:
            # 执行工具调用
            result = self.tools[tool_name](chunk, **merged_params)
            
            # 记录这一步的输入输出
            step_number = len(os.listdir(self.run_log_dir)) + 1
            
            self.log_strategy_step(
                strategy_name=tool_name,
                step_number=step_number,
                input_data={
                    "chunk": chunk,
                    "params": merged_params
                },
                output_data={"result": result}
            )
            
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution error: {error_msg}")
            
            # 记录错误
            step_number = len(os.listdir(self.run_log_dir)) + 1
            
            self.log_strategy_step(
                strategy_name=tool_name,
                step_number=step_number,
                input_data={
                    "chunk": chunk,
                    "params": merged_params
                },
                output_data=None,
                error=error_msg
            )
            
            raise

    def read_system_prompt(self, prompt_name: str) -> str:
        """Read system prompt from file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, 'patterns', prompt_name, 'system.md')
        
        try:
            with open_file_utf8(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading system prompt: {e}")
            raise

# Strategy Implementation
class ProcessingStrategy:
    """Implements a single processing strategy"""
    def __init__(self, config: Dict[str, Any]):
        self.config = StrategyConfig(**config)
        self.tool_params = self.config.tool_params or {}
        if self.config.input_format and self.config.prompt_name:
            self.user_input_template = self.config.input_format
        else:
            self.user_input_template = "{{text}}"

    def process(self, chunk: str, processor: TextProcessor, previous_outputs: Dict[str, str]) -> str:
        """Execute the strategy on a text chunk"""
        logger.debug(f"Processing strategy: {self.config.prompt_name or self.config.tool_name}")
        
        try:
            if self.config.tool_name:
                result = processor.execute_tool(self.config.tool_name, chunk, self.tool_params)
            elif self.config.model:
                prompt = self.prepare_prompt(chunk, processor, previous_outputs)
                result = processor.execute_model(self.config.model, prompt)
            else:
                raise ValueError("Neither tool_name nor model specified in strategy")
            
            return result if result is not None else ""
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            return ""

    def prepare_prompt(self, chunk: str, processor: TextProcessor, previous_outputs: Dict[str, str]) -> Union[str, List[Dict[str, str]]]:
        """Prepare prompt for model-based strategy"""
        # If chunk is already in messages format, return it directly
        if isinstance(chunk, list) and all(isinstance(m, dict) for m in chunk):
            return chunk
        
        # Format user input using template
        user_input = self.user_input_template
        
        # Replace previous outputs
        for key, value in previous_outputs.items():
            user_input = user_input.replace(f"{{{{{key}}}}}", str(value))
        
        # Replace current chunk
        user_input = user_input.replace("{{text}}", chunk)
        
        # Replace memory placeholders
        for key, value in processor.memory.items():
            user_input = user_input.replace(f"{{{{memory_{key}}}}}", str(value))

        return user_input

# Search Tools Implementation
class SearchTools:
    """Implementation of search-related tools"""
    @staticmethod
    def exa_search(query: str, **kwargs) -> str:
        """Execute search using Exa API"""
        logger.info(f"Executing Exa search with query: {query}")
        exa = Exa(os.getenv("EXA_API_KEY"))
        
        query_lines = query.strip().split('\n')
        actual_query = query_lines[0]
        
        search_params = {
            "query": actual_query,
            "num_results": 10,
            "start_published_date": None,
            "use_autoprompt": True,
            "category": "tweet",
            "text": {"max_characters": 2000},
            "highlights": {
                "highlights_per_url": 2,
                "num_sentences": 1,
                "query": f"This is the highlight query: {actual_query}"
            }
        }
        
        # Update with provided parameters
        search_params.update(kwargs)
        
        # Handle date parameter
        if len(query_lines) > 1 and query_lines[1].isdigit() and len(query_lines[1]) == 8:
            date_str = query_lines[1]
            search_params['start_published_date'] = (
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T00:00:00.000Z"
            )
        elif search_params['start_published_date'] is None and search_params['category'] == "tweet":
            search_params['start_published_date'] = (
                datetime.now() - timedelta(hours=72)
            ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Remove None values
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        logger.info(f"Search parameters: {json.dumps(search_params, indent=2)}")
        
        try:
            results = exa.search_and_contents(**search_params)
            logger.info(f"Exa {search_params['category']} search completed")
            return f'# Topic: {actual_query}\n{str(results)}'
        except Exception as e:
            logger.error(f"Exa search error: {str(e)}")
            return f"Error in Exa search: {str(e)}"

# Output Management
def save_output(results: List[str], output_path: str, output_format: str):
    """Save processing results to file"""
    output_handlers = {
        "json": lambda data, file: json.dump(data, file, indent=2, ensure_ascii=False),
        "md": lambda data, file: file.write("\n\n".join(filter(None, data))),
        "txt": lambda data, file: file.write("\n\n".join(filter(None, data)))
    }
    
    with open_file_utf8(output_path, "w") as f:
        output_handlers[output_format](results, f)

# Main Application
def main():
    """Main application entry point"""
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Process text with configurable workflows.")
    parser.add_argument("input_file", type=str, help="Path to input text file")
    parser.add_argument("--workflow", type=str, required=True, help="Workflow type")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Maximum tokens per chunk")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--config", type=str, help="Custom config file path")
    parser.add_argument("--output_format", type=str, default="md",
                      choices=["md", "txt", "json"], help="Output format")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = ConfigManager.load_config(args.workflow, args.config)
        if args.debug:
            logger.info("Config loaded successfully")
            logger.info(f"Config: {json.dumps(config, indent=2)}")

        # Initialize processor
        processor = TextProcessor(config, args.max_tokens, args.verbose, args.debug)

        # Setup input/output paths
        input_path = Path(args.input_file).resolve()
        output_path = input_path.parent / f"{args.workflow}-output.{args.output_format}"

        # Process text
        with open_file_utf8(input_path, "r") as f:
            text = f.read()

        if args.debug:
            logger.info("Input text loaded successfully")

        results = processor.process_text(text)

        if not results:
            logger.warning("No results generated")
        else:
            # Save results
            save_output(results, output_path, args.output_format)
            if args.debug:
                logger.info(f"Results saved to {output_path}")

            # Copy to clipboard
            with open_file_utf8(output_path, "r") as f:
                content = f.read()
            pyperclip.copy(content)
            if args.debug:
                logger.info("Content copied to clipboard")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()