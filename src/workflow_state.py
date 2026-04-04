"""
Workflow State Management
Stores intermediate results and metadata between steps to minimize LLM context.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class WorkflowState:
    """
    Structured state object that holds workflow context.
    Replaces storing everything in LLM conversation history.
    """
    
    def __init__(self):
        self.dataset_info: Optional[Dict[str, Any]] = None
        self.profiling_summary: Optional[Dict[str, Any]] = None
        self.quality_issues: Optional[Dict[str, Any]] = None
        self.cleaning_results: Optional[Dict[str, Any]] = None
        self.feature_engineering: Optional[Dict[str, Any]] = None
        self.modeling_results: Optional[Dict[str, Any]] = None
        self.visualization_paths: List[str] = []
        self.current_file: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: Optional[str] = None  # 'classification', 'regression', etc.
        self.steps_completed: List[str] = []
        self.created_at = datetime.utcnow().isoformat()
    
    def update_dataset_info(self, info: Dict[str, Any]):
        """Store basic dataset metadata (schema, shape, etc.)"""
        self.dataset_info = info
        self.current_file = info.get('file_path')
        self.steps_completed.append('dataset_loaded')
    
    def update_profiling(self, summary: Dict[str, Any]):
        """Store profiling results summary"""
        self.profiling_summary = summary
        self.steps_completed.append('profiling_complete')
    
    def update_quality(self, issues: Dict[str, Any]):
        """Store data quality assessment"""
        self.quality_issues = issues
        self.steps_completed.append('quality_checked')
    
    def update_cleaning(self, results: Dict[str, Any]):
        """Store cleaning/preprocessing results"""
        self.cleaning_results = results
        if results.get('output_file'):
            self.current_file = results['output_file']
        self.steps_completed.append('data_cleaned')
    
    def update_features(self, results: Dict[str, Any]):
        """Store feature engineering results"""
        self.feature_engineering = results
        if results.get('output_file'):
            self.current_file = results['output_file']
        self.steps_completed.append('features_engineered')
    
    def update_modeling(self, results: Dict[str, Any]):
        """Store model training results"""
        self.modeling_results = results
        self.steps_completed.append('model_trained')
    
    def add_visualization(self, path: str):
        """Track generated visualization"""
        self.visualization_paths.append(path)
    
    def get_context_for_step(self, step_name: str) -> Dict[str, Any]:
        """
        Get minimal context needed for a specific step.
        This replaces sending full conversation history to LLM.
        """
        context = {
            'current_file': self.current_file,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'steps_completed': self.steps_completed
        }
        
        # Step-specific context slicing
        if step_name == 'profiling':
            context['dataset_info'] = self.dataset_info
            
        elif step_name == 'quality_check':
            context['dataset_info'] = self.dataset_info
            context['profiling'] = self.profiling_summary
            
        elif step_name == 'cleaning':
            context['quality_issues'] = self.quality_issues
            context['profiling'] = self.profiling_summary
            
        elif step_name == 'feature_engineering':
            context['cleaning_results'] = self.cleaning_results
            context['dataset_info'] = self.dataset_info
            
        elif step_name == 'modeling':
            context['feature_engineering'] = self.feature_engineering
            context['cleaning_results'] = self.cleaning_results
            context['target_column'] = self.target_column
            context['task_type'] = self.task_type
            
        elif step_name == 'visualization':
            context['modeling_results'] = self.modeling_results
            context['dataset_info'] = self.dataset_info
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for storage/debugging"""
        return {
            'dataset_info': self.dataset_info,
            'profiling_summary': self.profiling_summary,
            'quality_issues': self.quality_issues,
            'cleaning_results': self.cleaning_results,
            'feature_engineering': self.feature_engineering,
            'modeling_results': self.modeling_results,
            'visualization_paths': self.visualization_paths,
            'current_file': self.current_file,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'steps_completed': self.steps_completed,
            'created_at': self.created_at
        }
    
    def save_to_file(self, path: str):
        """Save state to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: str) -> 'WorkflowState':
        """Load state from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        state = cls()
        state.dataset_info = data.get('dataset_info')
        state.profiling_summary = data.get('profiling_summary')
        state.quality_issues = data.get('quality_issues')
        state.cleaning_results = data.get('cleaning_results')
        state.feature_engineering = data.get('feature_engineering')
        state.modeling_results = data.get('modeling_results')
        state.visualization_paths = data.get('visualization_paths', [])
        state.current_file = data.get('current_file')
        state.target_column = data.get('target_column')
        state.task_type = data.get('task_type')
        state.steps_completed = data.get('steps_completed', [])
        state.created_at = data.get('created_at')
        
        return state
