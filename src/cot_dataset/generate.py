"""Generate CoT dataset using async LiteLLM calls."""

import asyncio
import json
import os
import platform
import warnings
from datetime import datetime
from logging import getLogger

from dotenv import load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm

from cot_dataset.constants import (
    COT_DATASET_FILE,
    MAX_CONCURRENT_REQUESTS,
    OUTPUT_DIR,
    SAVE_EVERY,
    TARGET_STYLE,
    TEACHER_MODEL,
)
from cot_dataset.prompts import create_cot_prompt_few_shot
from dataset.constants import DATASET_SLICE_BY_SPLIT_TYPE
from dataset.enums import DatasetSplit, WNCColumn
from dataset.preprocess import get_preprocessed_dataset_slice

load_dotenv()
logger = getLogger(__name__)

# Windows async setup
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    warnings.filterwarnings("ignore", category=ResourceWarning)


async def generate_single_cot(
    biased_text: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate CoT response for a single biased text."""
    async with semaphore:
        try:
            prompt = create_cot_prompt_few_shot(biased_text, TARGET_STYLE)

            response = await acompletion(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "input": biased_text,
                "style": TARGET_STYLE,
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "model": TEACHER_MODEL,
            }
        except Exception as e:
            logger.error(
                f"Error generating CoT for text: {biased_text[:50]}... Error: {e}"
            )
            return None


async def generate_cot_dataset_async(
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
) -> list[dict]:
    """Generate CoT dataset for all entries in the given split."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing_inputs = set()
    existing_results = []
    if COT_DATASET_FILE.exists():
        with open(COT_DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                existing_inputs.add(entry["input"])
                existing_results.append(entry)
        logger.info(f"Loaded {len(existing_results)} existing entries")

    # Get WNC data
    slice_range = DATASET_SLICE_BY_SPLIT_TYPE[dataset_split]
    dataset = get_preprocessed_dataset_slice(slice_range)
    biased_texts = dataset[WNCColumn.BIASED]

    # Filter out already processed
    texts_to_process = [t for t in biased_texts if t not in existing_inputs]
    logger.info(
        f"Processing {len(texts_to_process)}/{len(biased_texts)} texts "
        f"(skipping {len(biased_texts) - len(texts_to_process)} already processed)"
    )

    if not texts_to_process:
        logger.info("All texts already processed!")
        return existing_results

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [generate_single_cot(text, api_key, semaphore) for text in texts_to_process]

    # Process with progress bar
    new_results = []
    completed_count = 0

    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generating CoT"
    ):
        result = await coro
        if result is not None:
            new_results.append(result)
            completed_count += 1

            # Periodic save
            if completed_count % SAVE_EVERY == 0:
                _save_results(existing_results + new_results)
                logger.info(
                    f"Checkpoint: Saved {len(existing_results) + len(new_results)} total entries"
                )

    # Final save
    all_results = existing_results + new_results
    _save_results(all_results)
    logger.info(f"Completed! Total entries: {len(all_results)}")

    return all_results


def _save_results(results: list[dict]) -> None:
    """Save results to JSONL file."""
    with open(COT_DATASET_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_cot_dataset(
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
) -> list[dict]:
    """Synchronous wrapper for async generation."""
    if platform.system() == "Windows":
        return asyncio.run(generate_cot_dataset_async(dataset_split, max_concurrent))
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                generate_cot_dataset_async(dataset_split, max_concurrent)
            )
        finally:
            loop.close()
