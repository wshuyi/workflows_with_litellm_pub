import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
import pyperclip
from typing import List, Dict, Any, Union, Callable
from tqdm import tqdm
import sys
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion
import yaml
import re
import logging
from contextlib import contextmanager
import time
from datetime import datetime, timezone, timedelta
from exa_py import Exa

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'text_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Text processing script started")

class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        if 'strategies' not in config:
            raise ValueError("Configuration must contain 'strategies' key")
        
        for i, strategy in enumerate(config['strategies']):
            if 'tool_name' in strategy:
                required_keys = ['tool_name', 'input_format', 'output_name']
            elif 'model' in strategy:
                required_keys = ['model', 'prompt_name', 'input_format', 'output_name']
            else:
                raise ValueError(f"Strategy {i} must contain either 'tool_name' or 'model'")
            
            for key in required_keys:
                if key not in strategy:
                    raise ValueError(f"Strategy {i} is missing required key: {key}")

class APIClient:
    @staticmethod
    def query_api(messages: List[Dict[str, str]], model: str) -> str:
        if model.startswith("openai/") or model.find("/") == -1:
            load_dotenv(override=True)
            base_url = os.getenv("OPENAI_API_BASE")
            api_key = os.getenv("OPENAI_API_KEY")
            kwargs = {"api_key": api_key, "base_url": base_url}
            logger.info(f"Using OpenAI API with base URL: {base_url}")
        else:
            kwargs = {}
        
        try:
            response = completion(model=model, messages=messages, temperature=0.1, **kwargs)
            answer = response.choices[0].message.content
            logger.debug(f"API response: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            raise

class BaseTextProcessor:
    def __init__(self, max_tokens_per_chunk: int = 1000):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        self.config = {}
        self.chunk_count = 0
        self.current_chunk_number = 0
        self.memory = {}

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
                      lines[i-1].strip()[-1] not in '.!?:;'):
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
        
        for sep in separators:
            count = preprocessed_text.count(sep)
            logger.info(f"Separator '{sep}' appears {count} times")
        
        chunks = text_splitter.split_text(preprocessed_text)
        
        logger.info(f"Number of chunks after initial split: {len(chunks)}")
        
        if len(chunks) == 1:
            logger.warning("Only one chunk created. Printing chunk info:")
            logger.info(f"Chunk length: {len(chunks[0])}")
            logger.info(f"Chunk token count: {len(self.encoding.encode(chunks[0]))}")
        
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

    @contextmanager
    def open_file(self, file_path: str, mode: str):
        try:
            file = open(file_path, mode, encoding='utf-8')
            yield file
        finally:
            file.close()

    def read_system_prompt(self, pattern_name: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(current_dir, 'patterns', pattern_name, 'system.md')
        try:
            with self.open_file(system_prompt_path, 'r') as f:
                return f.read().strip()
        except IOError as e:
            logger.error(f"Error reading system prompt: {e}")
            raise

    def process_chunk(self, chunk: str) -> str:
        raise NotImplementedError("Subclasses must implement process_chunk method")

    def process_text(self, text: str) -> List[str]:
        chunks = self.split_text(text)
        self.chunk_count = len(chunks)
        logger.info(f"Text split into {self.chunk_count} chunks.")
        return [self.process_chunk(chunk) for chunk in tqdm(chunks, desc="Processing chunks")]

    def set_config(self, key: str, value: Any):
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def load_memory_files(self):
        memory_dir = os.path.join(current_dir, 'memory')
        if not os.path.exists(memory_dir):
            logger.warning(f"Memory directory not found: {memory_dir}")
            return

        for file_name in os.listdir(memory_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(memory_dir, file_name)
                key = file_name[:-3]
                try:
                    with self.open_file(file_path, 'r') as f:
                        self.memory[key] = f.read().strip()
                    logger.info(f"Loaded memory file: {file_name}")
                except IOError as e:
                    logger.error(f"Error reading memory file {file_name}: {e}")

class StrategyConfig:
    def __init__(self, tool_name: str = None, model: str = None, prompt_name: str = None, input_format: str = None, output_name: str = None):
        self.tool_name = tool_name
        self.model = model
        self.prompt_name = prompt_name
        self.input_format = input_format
        self.output_name = output_name

class ProcessingStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def _format_input(self, input_format: str, chunk: str, previous_outputs: Dict[str, str], original_text: str) -> str:
        formatted_input = input_format.replace("{{text}}", original_text)
        for key, value in previous_outputs.items():
            formatted_input = formatted_input.replace(f"{{{{{key}}}}}", value)
        return formatted_input

    def process(self, chunk: str, processor: 'TextProcessor', previous_outputs: Dict[str, str], original_text: str) -> str:
        formatted_input = self._format_input(self.config.input_format, chunk, previous_outputs, original_text)
        
        if self.config.tool_name:
            logger.info(f"Processing tool: {self.config.tool_name}")
            result = processor.execute_tool(self.config.tool_name, formatted_input)
        else:
            system_prompt = processor.read_system_prompt(self.config.prompt_name)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_input}
            ]
            logger.info(f"Processing strategy: {self.config.prompt_name}")
            result = APIClient.query_api(messages, self.config.model)
        return result

class TextProcessor(BaseTextProcessor):
    def __init__(self, strategies: List[ProcessingStrategy], max_tokens_per_chunk: int = 1000, verbose: bool = False):
        super().__init__(max_tokens_per_chunk)
        self.strategies = strategies
        self.verbose = verbose
        self.tools = {
            "search_exa": self.search_exa,
            "exa_paper_search": self.exa_paper_search
        }
        self.load_memory_files()
        load_dotenv()

    def process_chunk(self, chunk: str) -> str:
        self.current_chunk_number += 1
        print(f"\nProcessing chunk {self.current_chunk_number}/{self.chunk_count}")
        logger.info(f"Processing chunk {self.current_chunk_number}/{self.chunk_count}")
        result = chunk
        previous_outputs = {}
        original_text = chunk  # Store the original input text
        for i, strategy in enumerate(self.strategies, 1):
            print(f"Applying strategy {i}...")
            logger.info(f"Applying strategy {i}...")
            start_time = time.time()
            result = strategy.process(result, self, previous_outputs, original_text)
            end_time = time.time()
            previous_outputs[strategy.config.output_name] = result
            processing_time = end_time - start_time
            print(f"Strategy {i} completed in {processing_time:.2f} seconds")
            if self.verbose:
                print(f"Full result of strategy {i}:\n{result}")
            else:
                print(f"Result of strategy {i} (first 100 characters):")
                print(result[:100] + "..." if len(result) > 100 else result)
            
            logger.info(f"Strategy {i} completed in {processing_time:.2f} seconds")
            logger.info(f"Full result of strategy {i}:\n{result}")
        print(f"Chunk {self.current_chunk_number} processing complete")
        logger.info(f"Chunk {self.current_chunk_number} processing complete")
        return result

    def execute_tool(self, tool_name: str, input_text: str) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        params = self.parse_multi_input(input_text)
        if isinstance(params, str):
            # Single parameter case
            return self.tools[tool_name](params)
        else:
            # Multiple parameters case
            return self.tools[tool_name](**params)

    def parse_multi_input(self, input_text: str) -> Union[str, Dict[str, str]]:
        input_text = input_text.strip()
        if '\n' not in input_text and ':' not in input_text:
            # Single line input without key-value pair
            return input_text

        params = {}
        lines = input_text.split('\n')
        current_param = None
        current_value = []

        for line in lines:
            if ':' in line and not current_param:
                key, value = line.split(':', 1)
                current_param = key.strip()
                current_value = [value.strip()]
            elif current_param:
                if ':' in line:
                    params[current_param] = '\n'.join(current_value).strip()
                    key, value = line.split(':', 1)
                    current_param = key.strip()
                    current_value = [value.strip()]
                else:
                    current_value.append(line.strip())

        if current_param:
            params[current_param] = '\n'.join(current_value).strip()
        print(params)
        return params

    def search_exa(self, query: str) -> str:
        logger.info(f"Executing Exa search with query: {query}")
        exa = Exa(os.getenv("EXA_API_KEY"))
        start_date = (datetime.now() - timedelta(hours=72)).strftime("%Y-%m-%d")
        try:
            results = exa.search_and_contents(
                query,
                num_results=10,
                start_published_date=start_date,
                use_autoprompt=True,
                category="tweet",
                text={"max_characters": 2000},
                highlights={"highlights_per_url": 2, "num_sentences": 1, "query": f"This is the highlight query: {query}"}
            )
            
            logger.info("Exa search completed successfully")
            return f'# Topic: {query}\n{str(results)}'
        except Exception as e:
            logger.error(f"Error in Exa search: {str(e)}")
            return f"Error in Exa search: {str(e)}"

    def exa_paper_search(self, query: str, start_year: str) -> str:
        logger.info(f"Executing Exa paper search with query: {query} and start year: {start_year}")
        date = datetime(int(start_year), 1, 1, tzinfo=timezone.utc)
        start_published_date = date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        try:
            result = exa.search_and_contents(
                query,
                type="neural",
                use_autoprompt=True,
                num_results=10,
                text={
                    "max_characters": 1000
                },
                category="research paper",
                start_published_date=start_published_date,
                highlights={
                    "num_sentences": 3,
                    "highlights_per_url": 3
                }
            )
            logger.info("Exa paper search completed successfully")
            return f"Search Results for query '{query}' from year {start_year}:\n\n{str(result)}"
        except Exception as e:
            logger.error(f"Error in Exa paper search: {str(e)}")
            return f"Error in Exa paper search: {str(e)}"

class ConfigManager:
    @staticmethod
    def load_config(workflow_type: str) -> Dict[str, Any]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', f'{workflow_type}_config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            ConfigValidator.validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading or validating configuration: {e}")
            raise

class ProcessorFactory:
    @staticmethod
    def create_processor(workflow_type: str, max_tokens: int, verbose: bool) -> TextProcessor:
        config = ConfigManager.load_config(workflow_type)
        strategies = []
        for strategy_config in config['strategies']:
            strategy_obj = ProcessingStrategy(StrategyConfig(
                tool_name=strategy_config.get('tool_name'),
                model=strategy_config.get('model'),
                prompt_name=strategy_config.get('prompt_name'),
                input_format=strategy_config.get('input_format'),
                output_name=strategy_config['output_name']
            ))
            strategies.append(strategy_obj)
        return TextProcessor(strategies, max_tokens, verbose)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Process text with configurable workflows.")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("--workflow", type=str, required=True, help="Workflow type")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum tokens per chunk")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console output")
    args = parser.parse_args()

    input_path = Path(args.input_file).resolve()
    output_path = input_path.parent / f"{args.workflow}-output.md"

    try:
        processor = ProcessorFactory.create_processor(args.workflow, args.max_tokens, args.verbose)

        with processor.open_file(input_path, "r") as f:
            text = f.read()

        results = processor.process_text(text)

        with processor.open_file(output_path, "w") as f:
            f.write("\n\n".join(results))

        print(f"\nAll chunks processed. Results saved to {output_path}")
        logger.info(f"All chunks processed. Results saved to {output_path}")

        with processor.open_file(output_path, "r") as f:
            content = f.read()
        pyperclip.copy(content)
        print("Content copied to clipboard.")
        logger.info("Content copied to clipboard.")

        print("\nFinal processed text:")
        print("="*50)
        print(content)
        print("="*50)
        logger.info("Final processed text:\n" + "="*50 + "\n" + content + "\n" + "="*50)

    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()