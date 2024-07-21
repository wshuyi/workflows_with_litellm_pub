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
from datetime import datetime, timedelta
from exa_py import Exa

# 设置日志文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'text_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # 这会同时将日志输出到控制台
    ]
)

logger = logging.getLogger(__name__)

# 记录脚本开始执行的信息
logger.info("Text processing script started")

class ConfigValidator:
    """Validates the configuration file for the text processing workflow."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validates the configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
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
    """Handles API calls to language models."""

    @staticmethod
    def query_api(messages: List[Dict[str, str]], model: str) -> str:
        """
        Queries the API with the given messages and model.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the API.
            model (str): The model to use for the API call.

        Returns:
            str: The response from the API.

        Raises:
            Exception: If an error occurs during the API call.
        """
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
    """Base class for text processing operations."""

    def __init__(self, max_tokens_per_chunk: int = 1000):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        self.config = {}
        self.chunk_count = 0
        self.current_chunk_number = 0
        self.memory = {}  # 新增：用于存储 memory 文件内容

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by joining lines within paragraphs.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
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
        
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " ", ""]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=20,
            length_function=lambda t: len(self.encoding.encode(t)),
            separators=separators
        )
        
        # 打印每个分隔符的出现次数
        for sep in separators:
            count = preprocessed_text.count(sep)
            logger.info(f"Separator '{sep}' appears {count} times")
        
        chunks = text_splitter.split_text(preprocessed_text)
        
        logger.info(f"Number of chunks after initial split: {len(chunks)}")
        
        # 如果只有一个块，打印更多信息
        if len(chunks) == 1:
            logger.warning("Only one chunk created. Printing chunk info:")
            logger.info(f"Chunk length: {len(chunks[0])}")
            logger.info(f"Chunk token count: {len(self.encoding.encode(chunks[0]))}")
        
        result = [self._split_chunk(chunk) if len(self.encoding.encode(chunk)) > self.max_tokens_per_chunk else chunk for chunk in chunks]
        
        logger.info(f"Final number of chunks: {len(result)}")
        
        return result

    def _split_chunk(self, chunk: str) -> str:
        """
        Splits a chunk into sub-chunks if it exceeds the maximum token limit.

        Args:
            chunk (str): The chunk to split.

        Returns:
            str: The split chunk.
        """
        sentences = re.split(r'(?<=[。！？.!?])\s*', chunk)
        sub_chunks = []
        current_sub_chunk = []
        current_token_count = 0
        for sentence in sentences:
            sentence_token_count = len(self.encoding.encode(sentence))
            if sentence_token_count > self.max_tokens_per_chunk:
                # 如果单个句子超过最大 token 数，强制分割
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
        """
        Context manager for safely opening and closing files.

        Args:
            file_path (str): The path to the file.
            mode (str): The mode in which to open the file.

        Yields:
            file object: The opened file object.
        """
        try:
            file = open(file_path, mode, encoding='utf-8')
            yield file
        finally:
            file.close()

    def read_system_prompt(self, pattern_name: str) -> str:
        """
        Reads the system prompt from a file.

        Args:
            pattern_name (str): The name of the prompt pattern.

        Returns:
            str: The content of the system prompt.

        Raises:
            IOError: If there's an error reading the file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(current_dir, 'patterns', pattern_name, 'system.md')
        try:
            with self.open_file(system_prompt_path, 'r') as f:
                return f.read().strip()
        except IOError as e:
            logger.error(f"Error reading system prompt: {e}")
            raise

    def process_chunk(self, chunk: str) -> str:
        """
        Processes a single chunk of text. To be implemented by subclasses.

        Args:
            chunk (str): The chunk of text to process.

        Returns:
            str: The processed chunk of text.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement process_chunk method")

    def process_text(self, text: str) -> List[str]:
        """
        Processes the entire input text by splitting it into chunks and processing each chunk.

        Args:
            text (str): The input text to process.

        Returns:
            List[str]: A list of processed text chunks.
        """
        chunks = self.split_text(text)
        self.chunk_count = len(chunks)
        logger.info(f"Text split into {self.chunk_count} chunks.")
        return [self.process_chunk(chunk) for chunk in tqdm(chunks, desc="Processing chunks")]

    def set_config(self, key: str, value: Any):
        """
        Sets a configuration value.

        Args:
            key (str): The configuration key.
            value (Any): The configuration value.
        """
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value.

        Args:
            key (str): The configuration key.
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The configuration value.
        """
        return self.config.get(key, default)

    def load_memory_files(self):
        """
        Loads all .md files from the memory directory and stores their content in self.memory.
        """
        memory_dir = os.path.join(current_dir, 'memory')
        if not os.path.exists(memory_dir):
            logger.warning(f"Memory directory not found: {memory_dir}")
            return

        for file_name in os.listdir(memory_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(memory_dir, file_name)
                key = file_name[:-3]  # Remove '.md' from the file name
                try:
                    with self.open_file(file_path, 'r') as f:
                        self.memory[key] = f.read().strip()
                    logger.info(f"Loaded memory file: {file_name}")
                except IOError as e:
                    logger.error(f"Error reading memory file {file_name}: {e}")

class StrategyConfig:
    """Configuration for a processing strategy."""

    def __init__(self, tool_name: str = None, model: str = None, prompt_name: str = None, input_format: str = None, output_name: str = None):
        self.tool_name = tool_name
        self.model = model
        self.prompt_name = prompt_name
        self.input_format = input_format
        self.output_name = output_name

class ProcessingStrategy:
    """Represents a single processing strategy in the workflow."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def _format_input(self, original_text: str, current_text: str, previous_outputs: Dict[str, str], memory: Dict[str, str]) -> str:
        """
        Formats the input for the current strategy based on the input format, previous outputs, and memory.

        Args:
            original_text (str): The original unprocessed text.
            current_text (str): The current state of the text being processed.
            previous_outputs (Dict[str, str]): The outputs from previous strategies.
            memory (Dict[str, str]): The memory files content.

        Returns:
            str: The formatted input text for the current strategy.
        """
        input_text = self.config.input_format
        input_text = input_text.replace("{{text}}", original_text)
        for key, value in previous_outputs.items():
            input_text = re.sub(r'\{\{' + re.escape(key) + r'\}\}', value, input_text)
        for key, value in memory.items():
            input_text = re.sub(r'\{\{' + re.escape(key) + r'\}\}', value, input_text)
        return input_text

    def process(self, original_text: str, current_text: str, processor: 'TextProcessor', previous_outputs: Dict[str, str]) -> str:
        """
        Processes the text using the current strategy.

        Args:
            original_text (str): The original unprocessed text.
            current_text (str): The current state of the text being processed.
            processor (TextProcessor): The text processor instance.
            previous_outputs (Dict[str, str]): The outputs from previous strategies.

        Returns:
            str: The processed text after applying the current strategy.
        """
        if self.config.tool_name:
            input_text = self._format_input(original_text, current_text, previous_outputs, processor.memory)
            logger.info(f"Processing tool: {self.config.tool_name}")
            result = processor.execute_tool(self.config.tool_name, input_text)
        else:
            system_prompt = processor.read_system_prompt(self.config.prompt_name)
            input_text = self._format_input(original_text, current_text, previous_outputs, processor.memory)
            messages = [
                {"role": "system", "content": system_prompt},
{"role": "user", "content": input_text}
            ]
            logger.info(f"Processing strategy: {self.config.prompt_name}")
            result = APIClient.query_api(messages, self.config.model)
        return result

class TextProcessor(BaseTextProcessor):
    """Main text processor that applies a series of processing strategies."""

    def __init__(self, strategies: List[ProcessingStrategy], max_tokens_per_chunk: int = 1000, verbose: bool = False):
        super().__init__(max_tokens_per_chunk)
        self.strategies = strategies
        self.verbose = verbose
        self.tools = {
            "search_exa": self.search_exa
        }
        self.load_memory_files()  # 初始化时加载 memory 文件

    def process_chunk(self, chunk: str) -> str:
        """
        Processes a single chunk of text by applying all strategies in sequence.

        Args:
            chunk (str): The chunk of text to process.

        Returns:
            str: The processed chunk of text.
        """
        self.current_chunk_number += 1
        print(f"\nProcessing chunk {self.current_chunk_number}/{self.chunk_count}")
        logger.info(f"Processing chunk {self.current_chunk_number}/{self.chunk_count}")
        original_chunk = chunk
        result = chunk
        previous_outputs = {}
        for i, strategy in enumerate(self.strategies, 1):
            print(f"Applying strategy {i}...")
            logger.info(f"Applying strategy {i}...")
            start_time = time.time()
            result = strategy.process(original_chunk, result, self, previous_outputs)
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
        """
        Executes a specified tool with the given input text.

        Args:
            tool_name (str): The name of the tool to execute.
            input_text (str): The input text for the tool.

        Returns:
            str: The result of the tool execution.

        Raises:
            ValueError: If the specified tool is not found.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self.tools[tool_name](input_text)

    def search_exa(self, query: str) -> str:
        """
        Performs a search using the Exa API.

        Args:
            query (str): The search query.

        Returns:
            str: The raw search results from Exa API.
        """
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

class ConfigManager:
    """Manages loading and validating configuration files."""

    @staticmethod
    def load_config(workflow_type: str) -> Dict[str, Any]:
        """
        Loads and validates the configuration for a given workflow type.

        Args:
            workflow_type (str): The type of workflow to load the configuration for.

        Returns:
            Dict[str, Any]: The loaded and validated configuration.

        Raises:
            Exception: If there's an error loading or validating the configuration.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', f'{workflow_type}_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            ConfigValidator.validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading or validating configuration: {e}")
            raise

class ProcessorFactory:
    """Factory class for creating TextProcessor instances based on configuration."""

    @staticmethod
    def create_processor(workflow_type: str, max_tokens: int, verbose: bool) -> TextProcessor:
        """
        Creates a TextProcessor instance based on the given workflow type and maximum tokens.

        Args:
            workflow_type (str): The type of workflow to create a processor for.
            max_tokens (int): The maximum number of tokens per chunk.
            verbose (bool): Whether to enable verbose output.

        Returns:
            TextProcessor: An instance of TextProcessor configured for the specified workflow.
        """
        config = ConfigManager.load_config(workflow_type)
        strategies = []
        for strategy_config in config['strategies']:
            strategy_obj = ProcessingStrategy(StrategyConfig(
                tool_name=strategy_config.get('tool_name'),
                model=strategy_config.get('model'),
                prompt_name=strategy_config.get('prompt_name'),
                input_format=strategy_config['input_format'],
                output_name=strategy_config['output_name']
            ))
            strategies.append(strategy_obj)
        return TextProcessor(strategies, max_tokens, verbose)

def main():
    """
    Main function to run the text processing workflow.
    """
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