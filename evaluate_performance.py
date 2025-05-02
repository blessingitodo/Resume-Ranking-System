import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from resume_ranker import EnhancedResumeRanker

def calculate_precision_at_k(relevant_items, recommended_items, k):
    """Calculate precision@k metric"""
    if len(recommended_items) == 0 or k == 0:
        return 0.0
        
    # Get top-k recommended items
    top_k_items = recommended_items[:k]
    
    # Count relevant items in top-k
    num_relevant = len(set(top_k_items) & set(relevant_items))
    
    return num_relevant / min(k, len(recommended_items))

def calculate_recall_at_k(relevant_items, recommended_items, k):
    """Calculate recall@k metric"""
    if len(relevant_items) == 0 or len(recommended_items) == 0 or k == 0:
        return 0.0
        
    # Get top-k recommended items
    top_k_items = recommended_items[:k]
    
    # Count relevant items in top-k
    num_relevant = len(set(top_k_items) & set(relevant_items))
    
    return num_relevant / len(relevant_items)

# 3. Run the system on sample data
def run_ranking_system(job_descriptions, resume_texts):
    """Run the resume ranking system on sample data"""
    ranker = EnhancedResumeRanker()
    system_rankings = {}
    
    # For each job, rank all resumes
    for job_id, job_description in job_descriptions.items():
        # In a real scenario, you'd use actual files
        # Here we'll create a wrapper to simulate file behavior
        
        # First save resume texts to temporary files
        temp_files = []
        for resume_id, resume_text in resume_texts.items():
            file_path = f"temp_{resume_id}.txt"
            with open(file_path, 'w') as f:
                f.write(resume_text)
            temp_files.append(file_path)
        
        # Rank the resumes
        results = ranker.rank_resumes(job_description, temp_files)
        
        # Extract resume IDs from file names
        ranked_ids = [os.path.basename(result['file_name']).replace('temp_', '').replace('.txt', '') 
                     for result in results]
        
        system_rankings[job_id] = ranked_ids
        
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return system_rankings

# 4. Evaluate performance and generate charts
def evaluate_performance(ground_truth, system_rankings):
    """Calculate performance metrics and generate charts"""
    # Calculate metrics by job type
    job_type_results = {
        'technical': {'precision@5': [], 'recall@10': []},
        'management': {'precision@5': [], 'recall@10': []},
        'entry_level': {'precision@5': [], 'recall@10': []}
    }
    
    overall_precision5 = []
    overall_recall10 = []
    
    for job_id, relevant_items in ground_truth.items():
        job_type = job_id.split('_')[0]  # Extract job type from ID
        recommended_items = system_rankings.get(job_id, [])
        
        # Calculate metrics
        p5 = calculate_precision_at_k(relevant_items, recommended_items, 5)
        r10 = calculate_recall_at_k(relevant_items, recommended_items, 10)
        
        # Store by job type
        if job_type in job_type_results:
            job_type_results[job_type]['precision@5'].append(p5)
            job_type_results[job_type]['recall@10'].append(r10)
        
        overall_precision5.append(p5)
        overall_recall10.append(r10)
    
    # Calculate averages
    for job_type in job_type_results:
        if job_type_results[job_type]['precision@5']:
            job_type_results[job_type]['precision@5'] = np.mean(job_type_results[job_type]['precision@5'])
            job_type_results[job_type]['recall@10'] = np.mean(job_type_results[job_type]['recall@10'])
    
    overall_results = {
        'precision@5': np.mean(overall_precision5),
        'recall@10': np.mean(overall_recall10)
    }
    
    print("Performance Results:")
    print(f"Overall Precision@5: {overall_results['precision@5']:.2%}")
    print(f"Overall Recall@10: {overall_results['recall@10']:.2%}")
    print("\nBy Job Type:")
    for job_type, metrics in job_type_results.items():
        print(f"{job_type.title()}: Precision@5 = {metrics['precision@5']:.2%}, Recall@10 = {metrics['recall@10']:.2%}")
    
    # Generate charts
    plot_performance_by_job_type(job_type_results, overall_results)
    
    return job_type_results, overall_results

# 5. Plotting functions
def plot_performance_by_job_type(results, overall_results):
    """Create bar chart of performance metrics by job type"""
    # Get job types with valid results only
    valid_job_types = []
    precision_values = []
    recall_values = []
    
    for jt, metrics in results.items():
        # Only include job types with numeric precision/recall values
        if isinstance(metrics['precision@5'], (int, float)) and isinstance(metrics['recall@10'], (int, float)):
            valid_job_types.append(jt)
            precision_values.append(metrics['precision@5'])
            recall_values.append(metrics['recall@10'])
    
    # Add overall values if they exist
    if isinstance(overall_results['precision@5'], (int, float)) and isinstance(overall_results['recall@10'], (int, float)):
        valid_job_types.append('Overall')
        precision_values.append(overall_results['precision@5'])
        recall_values.append(overall_results['recall@10'])
    
    # Check if we have any valid data to plot
    if not valid_job_types:
        print("No valid job types to plot. Skipping chart generation.")
        return
    
    # Set up plot
    x = np.arange(len(valid_job_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precision_values, width, label='Precision@5')
    rects2 = ax.bar(x + width/2, recall_values, width, label='Recall@10')
    
    # Add labels and title
    ax.set_ylabel('Score')
    ax.set_title('Ranking Performance by Job Type')
    ax.set_xticks(x)
    ax.set_xticklabels([jt.replace('_', ' ').title() for jt in valid_job_types])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0%}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('performance_by_job_type.png', dpi=300)
    plt.show()

# Main execution for real data evaluation
if __name__ == "__main__":
    import os
    import json
    import pandas as pd
    
    # Define paths to your real data
    JOB_DESCRIPTIONS_PATH = "C:/Users/itodo/Downloads/job_descriptions" # Folder with job description text files
    RESUME_CSV_PATH = "C:/Users/itodo/Downloads/Copy of resumesdata - clean.csv"  # Folder with resume files (PDF, DOCX, TXT)
    GROUND_TRUTH_PATH = "path/to/expert_rankings.json"  # Optional: JSON file with expert rankings

    
    print("Loading real data...")
    
    # Load job descriptions
    job_descriptions = {}
    for filename in os.listdir(JOB_DESCRIPTIONS_PATH):
        if filename.endswith(".txt"):
            job_id = filename.replace(".txt", "")
            with open(os.path.join(JOB_DESCRIPTIONS_PATH, filename), 'r', encoding='utf-8') as f:
                job_descriptions[job_id] = f.read()
    
    # Load ground truth (expert rankings) if available
    ground_truth = {}
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    
    print(f"Loaded {len(job_descriptions)} job descriptions")
    
    # Create the ranker
    ranker = EnhancedResumeRanker()
    
    print("\nRunning resume ranking system...")
    
    # Run ranking for each job
    system_rankings = {}
    for job_id, job_description in job_descriptions.items():
        print(f"Ranking resumes for job: {job_id}")
        
        # Use the dataset functionality instead of individual files
        results = ranker.rank_dataset_resumes(job_description, RESUME_CSV_PATH, "csv")
        
        # Extract resume IDs from results
        ranked_ids = [result['resume_id'] for result in results]
        
        system_rankings[job_id] = ranked_ids
        
        print(f"Ranked {len(ranked_ids)} resumes for {job_id}")
    
    # Save system rankings to file
    with open("system_rankings.json", 'w', encoding='utf-8') as f:
        json.dump(system_rankings, f, indent=2)
    
    print("\nSystem rankings saved to system_rankings.json")
    
    # Generate synthetic ground truth if none exists
    if not ground_truth:
        print("No ground truth found. Generating synthetic ground truth for testing...")
        ground_truth = {}
        
        for job_id, rankings in system_rankings.items():
            # Take top 5 results from system as "ground truth" 
            # (just for testing the evaluation pipeline)
            ground_truth[job_id] = rankings[:5]
            
            # Add some variation to make it interesting
            if len(rankings) > 7:
                # Swap a few positions to create differences
                ground_truth[job_id][0], ground_truth[job_id][2] = ground_truth[job_id][2], ground_truth[job_id][0]
                
                # Add a resume that's ranked lower by the system
                if len(rankings) > 10:
                    ground_truth[job_id][4] = rankings[9]
        
        print(f"Created synthetic ground truth for {len(ground_truth)} jobs")
        
        # Optionally save this synthetic ground truth
        with open("synthetic_ground_truth.json", 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)

    # If ground truth is available, evaluate performance
    if ground_truth:
        print("\nEvaluating performance...")
        
        # Calculate metrics by job type
        job_type_results = {
            'technical': {'precision@5': [], 'recall@10': []},
            'management': {'precision@5': [], 'recall@10': []},
            'entry_level': {'precision@5': [], 'recall@10': []}
        }
        
        overall_precision5 = []
        overall_recall10 = []
        
        for job_id, relevant_items in ground_truth.items():
            if job_id not in system_rankings:
                continue
                
            # Extract job type from ID
            job_type = 'technical'  # Default
            if '_' in job_id:
                prefix = job_id.split('_')[0]
                if prefix in job_type_results:
                    job_type = prefix
            
            recommended_items = system_rankings[job_id]
            
            # Calculate metrics
            p5 = calculate_precision_at_k(relevant_items, recommended_items, 5)
            r10 = calculate_recall_at_k(relevant_items, recommended_items, 10)
            
            # Store by job type
            job_type_results[job_type]['precision@5'].append(p5)
            job_type_results[job_type]['recall@10'].append(r10)
            
            overall_precision5.append(p5)
            overall_recall10.append(r10)
        
        
        # Calculate averages
        for job_type in job_type_results:
            if job_type_results[job_type]['precision@5']:
                job_type_results[job_type]['precision@5'] = np.mean(job_type_results[job_type]['precision@5'])
                job_type_results[job_type]['recall@10'] = np.mean(job_type_results[job_type]['recall@10'])
        
        overall_results = {
            'precision@5': np.mean(overall_precision5) if overall_precision5 else 0,
            'recall@10': np.mean(overall_recall10) if overall_recall10 else 0
        }
        
        # Print results
        print("\nPerformance Results:")
        print(f"Overall Precision@5: {overall_results['precision@5']:.2%}")
        print(f"Overall Recall@10: {overall_results['recall@10']:.2%}")
        print("\nBy Job Type:")
        for job_type, metrics in job_type_results.items():
            if isinstance(metrics['precision@5'], (int, float)):
                print(f"{job_type.title()}: Precision@5 = {metrics['precision@5']:.2%}, Recall@10 = {metrics['recall@10']:.2%}")
        
        # Generate charts
        plot_performance_by_job_type(job_type_results, overall_results)
        print("\nEvaluation complete. Charts saved to current directory.")
    else:
        print("\nNo ground truth data available. Skipping performance evaluation.")