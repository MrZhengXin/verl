# Copyright 2025 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures  # <-- Import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3  # Number of retries for API calls
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 30

logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = ["python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp", "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl", "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script", "jest", "verilog", "python_gpu", "lean", "swift", "racket"]


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    in_outs: any,
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
    run_all_cases: bool = True,
    total_timeout: int = 30,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "completion": code,
            'config': {
                'language': 'python', 
                'compile_timeout': compile_timeout, 
                'run_timeout': run_timeout, 
                'memory_limit_mb': memory_limit_mb,
                'provided_data': {
                    'test_cases': in_outs
                }, 
                'extra': {
                    'run_all_cases': run_all_cases, 
                    'total_timeout': total_timeout
                }
            }
                            
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}")  # <-- Use internal log_prefix
            requests._client_max_size = 1000 * 1024 * 1024  # Set max request size to 100MB
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt {attempt + 1}/{MAX_RETRIES}"  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _process_single_case(
    case_index: int,
    in_outs: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
    continuous=False,
) -> Tuple[int, Dict[str, Any]]:
    """Helper function to process a single test case."""
    api_response = None
    error_msg = None
    logger.info(f"Processing test case {case_index + 1}.")

    current_generation_code = generation

    try:
        if concurrent_semaphore:
            logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
            with concurrent_semaphore:
                logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, in_outs=in_outs, compile_timeout=timeout, run_timeout=timeout, memory_limit_mb=memory_limit_mb, language=language)
            logger.debug(f"Case {case_index + 1}: Semaphore released.")
        else:
            api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, in_outs=in_outs, compile_timeout=timeout, run_timeout=timeout, memory_limit_mb=memory_limit_mb, language=language)
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        # traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": str(in_outs),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
        "score": 0.0,  # Default score, will be updated based on API response
    }
    result_status = -1  # Default error: API request error or unknown sandbox error

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1  # API request itself failed (includes timeout after retries)
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        # Log code and input only on error for brevity
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        # logger.error(f"Case {case_index}: input: {str(in_outs)}")
    elif api_response:
        # --- Add debug logging ---
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        # Extract compile information
        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        # Extract run information
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        # --- Determine status based on API response ---
        api_status = metadata["api_status"]

        # Note: Output might contain trailing newlines, need normalization
        if api_response.get("accepted", None) is True:
            result_status = True
            metadata["status"] = "success"
            metadata["score"] = 1.0  # Full score for correct answer
        else:
            result_status = False
            metadata["status"] = "wrong_answer"
            # Calculate score based on passed test cases ratio
            cases = api_response.get('tests', [])
            total_cases = len(cases)
            passed_cases = sum(1 for test in cases if test and test.get('passed', False))
            if total_cases == 0:
                result_status = -1
                passed_rate = 0
            else:
                if continuous:
                    passed_rate = passed_cases / total_cases
                else:
                    passed_rate = 1 if passed_cases == total_cases else 0
            metadata["score"] = passed_rate  # Score as a fraction of passed cases
    return result_status, metadata


def check_correctness_common_evaluate(
    sandbox_fusion_url: str, 
    in_outs: Optional[dict], 
    generation: str, 
    timeout: int = DEFAULT_TIMEOUT, 
    memory_limit_mb: int = 1024,
    language: str = "python", 
    concurrent_semaphore: Optional[threading.Semaphore] = None
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    logger.info("Starting correctness check for generation.")

    if not in_outs or "input" not in in_outs or "output" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    fn_name = in_outs.get("fn_name")
    num_cases = 1
    results = [None] * num_cases  # Initialize with placeholders
    metadata_list = [None] * num_cases  # Initialize with placeholders

    first_compile_error_index = -1

    # max_workers is limited by sandbox_fusion_max_concurrent from concurrent_semaphore
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        # Submit all tasks, passing the concurrent_semaphore to _process_single_case
        future_to_index = {executor.submit(_process_single_case, 0, in_outs, sandbox_fusion_url, generation, timeout, memory_limit_mb, language, concurrent_semaphore, fn_name): 0}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata

                # Check for compile error (-4)
                if result_status == -4:
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        first_compile_error_index = index
                    # Optimization: could potentially cancel futures for index > first_compile_error_index
                    # However, cancellation is not guaranteed. Post-processing is safer.

            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                # traceback.print_exc()
                results[index] = -1  # Mark as API/internal error
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(in_outs),
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Post-processing for compile errors
    if first_compile_error_index != -1:
        logger.warning(f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors.")
        for i in range(first_compile_error_index + 1, num_cases):
            # Only update if not already processed (though it should be None or have a result)
            if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                results[i] = -4
                # Update or create metadata for skipped cases due to compile error
                if metadata_list[i] is None:  # If future failed before returning metadata
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(in_outs),
                        "api_request_error": None,
                        "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                    }
                else:  # If future completed but result is overridden
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
