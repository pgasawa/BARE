"""Generate new math QA problems using BARE's existing GSM8K tasks."""

import os
import sys
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

# Add the bare directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
bare_root = os.path.join(current_dir, '..')
sys.path.insert(0, bare_root)

import src.logger as core
from experiments.generation_methods import GenerationMethod
from experiments.synthetic_data_runner import generate_data

# Import existing GSM8K tasks
try:
    from src.tasks.gsm8k_tasks import GSM8KDataGenerationTask
    print("✅ Successfully imported GSM8KDataGenerationTask")
except ImportError as e:
    print(f"❌ Error importing GSM8K tasks: {e}")
    sys.exit(1)


def generate_new_math_problems(
    num_problems: int = 10,
    generation_method: str = "instruct_fewshot",
    base_model: str = "gpt-4o-mini", 
    instruct_model: str = "gpt-4o",
    output_dir: str = "generated_math_qa"
):
    """Generate new math QA problems using BARE's GSM8K task.
    
    Args:
        num_problems: Number of problems to generate
        generation_method: Method to use (see available methods from explore_tasks.py)
        base_model: Base model for generation
        instruct_model: Instruct model for refinement
        output_dir: Directory to save results
    """
    
    print(f"🧮 Generating {num_problems} new math QA problems...")
    print(f"Method: {generation_method}")
    print(f"Base Model: {base_model}")
    print(f"Instruct Model: {instruct_model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the GSM8K task
    try:
        task = GSM8KDataGenerationTask()
        print("✅ Initialized GSM8KDataGenerationTask")
    except Exception as e:
        print(f"❌ Error initializing task: {e}")
        return None
    
    # Convert generation method string to enum
    try:
        method_enum = GenerationMethod(generation_method)
        print(f"✅ Using generation method: {method_enum}")
    except ValueError:
        print(f"❌ Invalid generation method: {generation_method}")
        print(f"Available methods: {[m.value for m in GenerationMethod]}")
        return None
    
    # Generate data
    start_time = time.time()
    
    try:
        print("🚀 Starting generation...")
        generated_data, total_cost = generate_data(
            num_calls=num_problems,
            examples_per_call=1,
            generation_method=method_enum,
            task=task,
            source_data=None,
            base_model=base_model,
            instruct_model=instruct_model,
        )
        print("✅ Generation completed")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    generation_time = time.time() - start_time
    
    print(f"✅ Generated {len(generated_data)} math problems")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"⏱️ Time taken: {generation_time:.2f} seconds")
    
    if not generated_data:
        print("⚠️ No data was generated. Check the logs for errors.")
        return None
    
    # Save results in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"math_qa_{generation_method}_{timestamp}"
    
    # 1. Save as pickle (BARE format)
    pickle_path = os.path.join(output_dir, f"{base_filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(generated_data, f)
    
    # 2. Save as JSON for easy inspection
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(generated_data, f, indent=2, default=str)
    
    # 3. Save human-readable format
    readable_path = os.path.join(output_dir, f"{base_filename}_readable.txt")
    with open(readable_path, 'w') as f:
        f.write(f"Generated Math QA Problems - {datetime.now()}\n")
        f.write(f"Method: {generation_method}\n")
        f.write(f"Base Model: {base_model}\n")
        f.write(f"Instruct Model: {instruct_model}\n")
        f.write(f"Total Problems: {len(generated_data)}\n")
        f.write(f"Cost: ${total_cost:.4f}\n")
        f.write(f"Time: {generation_time:.2f}s\n")
        f.write("="*80 + "\n\n")
        
        for i, problem in enumerate(generated_data, 1):
            f.write(f"PROBLEM {i}:\n")
            f.write(f"Question: {problem.get('question', problem.get('content', 'N/A'))}\n\n")
            f.write(f"Answer: {problem.get('answer', 'N/A')}\n")
            f.write("-"*60 + "\n\n")
    
    print(f"📁 Results saved to:")
    print(f"  - Pickle: {pickle_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Readable: {readable_path}")
    
    # Show preview of generated problems
    print(f"\n📋 Preview of generated problems:")
    for i, problem in enumerate(generated_data[:3], 1):
        print(f"\n--- Problem {i} ---")
        question = problem.get('question', problem.get('content', 'N/A'))
        answer = problem.get('answer', 'N/A')
        
        print(f"Q: {question}")
        if len(str(answer)) > 100:
            answer = str(answer)[:100] + "..."
        print(f"A: {answer}")
    
    return generated_data


if __name__ == "__main__":
    # Configuration - start with a small test
    config = {
        "num_problems": 5,  # Start very small for testing
        "generation_method": "instruct_fewshot",  # This should work well
        "base_model": "gpt-4o-mini",
        "instruct_model": "gpt-4o-mini",  # Use same model to save cost
        "output_dir": "generated_math_qa"
    }
    
    print("🚀 Starting Math QA Generation...")
    print(f"Config: {config}")
    
    result = generate_new_math_problems(**config)
    
    if result:
        print("🎉 Generation completed successfully!")
        print(f"Generated {len(result)} problems")
    else:
        print("❌ Generation failed. Check the output above for details.")

