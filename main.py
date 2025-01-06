from typing import List, Dict, Any
import json
import asyncio
import subprocess
from openai import AsyncAzureOpenAI
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'cora_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to execute AWS CLI commands
async def execute_aws_command(command: str) -> str:
    logger.info(f"Executing AWS CLI command: {command}")
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = f"Error: {stderr.decode()}"
            logger.error(f"AWS CLI command failed: {error_msg}")
            return error_msg
            
        success_output = stdout.decode()
        logger.info("AWS CLI command executed successfully")
        logger.debug(f"Command output: {success_output}")
        return success_output
    except Exception as e:
        logger.error(f"Exception in execute_aws_command: {str(e)}", exc_info=True)
        return f"Error executing command: {str(e)}"

# Define the functions that OpenAI can call
functions = [
    {
        "name": "execute_aws_cli",
        "description": "Execute AWS CLI commands to interact with AWS resources",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The AWS CLI command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "get_aws_cli_example",
        "description": "Retrieve example AWS CLI commands from the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for AWS CLI examples"
                }
            },
            "required": ["query"]
        }
    }
]

class CORA:
    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        deployment_name: str,
        knowledge_base: Dict[str, List[str]]
    ):
        logger.info("Initializing CORA")
        logger.info(f"Azure Endpoint: {azure_endpoint}")
        logger.info(f"API Version: {api_version}")
        logger.info(f"Deployment Name: {deployment_name}")
        
        # Validate endpoint format
        if not azure_endpoint.startswith('https://'):
            azure_endpoint = f'https://{azure_endpoint}'
        logger.info(f"Final Azure Endpoint: {azure_endpoint}")
        
        try:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                max_retries=3,
                timeout=30.0
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", exc_info=True)
            raise
            
        self.deployment_name = deployment_name
        self.knowledge_base = knowledge_base

    async def get_aws_cli_example(self, query: str) -> str:
        logger.info(f"Searching knowledge base for query: {query}")
        relevant_examples = []
        for key, examples in self.knowledge_base.items():
            if any(word.lower() in key.lower() for word in query.split()):
                relevant_examples.extend(examples)
                logger.debug(f"Found matching examples for key: {key}")
        
        result = "\n".join(relevant_examples) if relevant_examples else "No examples found"
        logger.info(f"Found {len(relevant_examples)} relevant examples")
        return result

    async def process_query(self, user_query: str) -> str:
        logger.info(f"Processing user query: {user_query}")
        try:
            # First, get relevant AWS CLI examples
            examples = await self.get_aws_cli_example(user_query)
            logger.info("Retrieved AWS CLI examples")
            
            messages = [
                {"role": "system", "content": "You are CORA, a Cloud Operations and Resource Assistant. You help users interact with AWS resources using AWS CLI commands."},
                {"role": "user", "content": f"Query: {user_query}\nRelevant AWS CLI examples:\n{examples}"}
            ]
            logger.debug(f"Prepared messages for OpenAI: {json.dumps(messages, indent=2)}")

            # Get completion with function calling
            logger.info(f"Sending request to OpenAI with deployment: {self.deployment_name}")
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                functions=functions,
                function_call="auto"
            )
            logger.info("Received response from OpenAI")
            logger.debug(f"OpenAI response: {response}")

            # Process the response
            assistant_message = response.choices[0].message
            logger.info("Processing assistant message")
            
            if assistant_message.function_call:
                function_name = assistant_message.function_call.name
                function_args = json.loads(assistant_message.function_call.arguments)
                logger.info(f"Function call detected: {function_name}")
                logger.debug(f"Function arguments: {function_args}")
                
                if function_name == "execute_aws_cli":
                    # Execute the AWS CLI command
                    result = await execute_aws_command(function_args["command"])
                    logger.info("AWS CLI command execution completed")
                    
                    # Get final response from OpenAI
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": function_name,
                                "arguments": json.dumps(function_args)
                            }
                        },
                        {
                            "role": "function",
                            "name": function_name,
                            "content": result
                        }
                    ])
                    
                    logger.info("Sending final request to OpenAI")
                    final_response = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages
                    )
                    logger.info("Received final response from OpenAI")
                    
                    return final_response.choices[0].message.content
                
            return assistant_message.content

        except Exception as e:
            logger.error("Error in process_query", exc_info=True)
            return f"Error processing query: {str(e)}"

async def main():
    logger.info("Starting CORA application")
    
    # Example knowledge base with AWS CLI commands
    knowledge_base = {
        "list ec2 instances": [
            "aws ec2 describe-instances",
            "aws ec2 describe-instances --filters Name=instance-state-name,Values=running"
        ],
        "create s3 bucket": [
            "aws s3 mb s3://bucket-name",
            "aws s3api create-bucket --bucket bucket-name --region region-name"
        ]
    }
    logger.info("Knowledge base initialized")

    try:
        logger.info("Creating CORA instance")
        cora = CORA(
            api_key="###",
            api_version="###",
            azure_endpoint="####",
            deployment_name="gpt-4o-mini",
            knowledge_base=knowledge_base
        )

        logger.info("Processing test query")
        result = await cora.process_query("List all running EC2 instances in mumbai")
        logger.info("Query processing completed")
        print(result)
        
    except Exception as e:
        logger.error("Main error", exc_info=True)
        print(f"Main error: {str(e)}")

if __name__ == "__main__":
    logger.info("Application starting")
    asyncio.run(main())
    logger.info("Application finished")
