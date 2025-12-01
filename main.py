"""
Main entry point for Agentic Gaming Analytics System
Runs the complete 5-agent pipeline with 3-layer guardrails
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
from loguru import logger

from src.orchestrator import AgenticOrchestrator
from src.utils.logger import setup_logger
from src.utils.data_loader import DataLoader

# Setup logging
setup_logger(log_level="INFO")


def run_full_demo():
    """Run a complete demo of the agentic system"""
    
    print("=" * 80)
    print("🎮 AGENTIC GAMING ANALYTICS SYSTEM")
    print("=" * 80)
    print("\n5-Agent System with 3-Layer Guardrails\n")
    print("Agents:")
    print("  1. Data Ingestion & Preprocessing")
    print("  2. Multi-Model Prediction (RF + XGBoost + NN)")
    print("  3. Prescriptive Strategy (RL Bandit)")
    print("  4. Execution & Simulation")
    print("  5. Monitoring & Adaptive Learning")
    print("\nGuardrails:")
    print("  Layer 1: Input Validation")
    print("  Layer 2: Prediction Validation")
    print("  Layer 3: Action Validation")
    print("\n" + "=" * 80 + "\n")
    
    # Initialize orchestrator
    logger.info("Initializing Agentic Orchestrator...")
    orchestrator = AgenticOrchestrator()
    
    # Load data
    logger.info("Loading gaming behavior dataset...")
    data_loader = DataLoader()
    
    try:
        df = data_loader.load_gaming_dataset()
        logger.info(f"✓ Loaded {len(df)} player records")
        
        # Display dataset info
        print("\n📊 DATASET OVERVIEW")
        print("-" * 80)
        print(f"Total Players: {len(df):,}")
        print(f"Features: {len(df.columns)}")
        print(f"\nEngagement Distribution:")
        engagement_dist = df['EngagementLevel'].value_counts()
        for level, count in engagement_dist.items():
            pct = count / len(df) * 100
            print(f"  {level}: {count:,} ({pct:.1f}%)")
        
        print(f"\nAge Range: {df['Age'].min()} - {df['Age'].max()}")
        print(f"Avg Playtime: {df['PlayTimeHours'].mean():.1f} hours")
        print(f"Top Genre: {df['GameGenre'].mode()[0]}")
        print()
        
    except FileNotFoundError:
        logger.error("Dataset not found! Please download the dataset first.")
        print("\n❌ Dataset not found!")
        print("\nPlease download from:")
        print("https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
        print("\nOr run: kaggle datasets download -d rabieelkharoua/predict-online-gaming-behavior-dataset")
        return
    
    # Prepare data for training
    logger.info("Preparing data for training...")
    from sklearn.model_selection import train_test_split
    from src.utils.feature_engineering import FeatureEngineer
    
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train the pipeline
    print("\n🤖 TRAINING MULTI-AGENT SYSTEM")
    print("-" * 80)
    
    train_results = orchestrator.train_pipeline(X_train, y_train)
    
    print("\n✓ Training complete!")
    print(f"\nModel Performance:")
    if 'metrics' in train_results:
        metrics = train_results['metrics']
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Precision: {metrics.get('weighted_precision', 0):.3f}")
        print(f"  Recall: {metrics.get('weighted_recall', 0):.3f}")
        print(f"  F1-Score: {metrics.get('weighted_f1', 0):.3f}")
    
    # Test on sample players
    print("\n\n🎯 PROCESSING SAMPLE PLAYERS")
    print("-" * 80)
    
    # Select diverse test samples
    test_indices = []
    for engagement_level in ['High', 'Medium', 'Low']:
        level_mask = df.loc[X_test.index, 'EngagementLevel'] == engagement_level
        level_indices = X_test.index[level_mask]
        if len(level_indices) > 0:
            test_indices.append(level_indices[0])
    
    # Process each sample
    for idx, test_idx in enumerate(test_indices[:3]):
        print(f"\n{'=' * 80}")
        print(f"PLAYER {idx + 1}")
        print(f"{'=' * 80}")
        
        player_features = X_test.loc[test_idx].values.reshape(1, -1)
        player_data = df.loc[test_idx].to_dict()
        
        # Display player profile
        print(f"\n📋 Player Profile:")
        print(f"  ID: {player_data.get('PlayerID', 'N/A')}")
        print(f"  Age: {player_data.get('Age', 'N/A')}")
        print(f"  Genre: {player_data.get('GameGenre', 'N/A')}")
        print(f"  Playtime: {player_data.get('PlayTimeHours', 0):.1f} hours")
        print(f"  Sessions/Week: {player_data.get('SessionsPerWeek', 0)}")
        print(f"  Level: {player_data.get('PlayerLevel', 0)}")
        print(f"  Actual Engagement: {player_data.get('EngagementLevel', 'N/A')}")
        
        # Process through pipeline
        result = orchestrator.process_player(
            player_features=player_features,
            player_data=player_data,
            execute_action=True
        )
        
        # Display results
        if result.get('success'):
            prediction_result = result.get('prediction', {})
            strategy_result = result.get('strategy', {})
            execution_result = result.get('execution', {})
            
            print(f"\n🔮 Prediction:")
            print(f"  Predicted Engagement: {prediction_result.get('prediction', 'N/A')}")
            print(f"  Confidence: {prediction_result.get('confidence', 0):.2%}")
            print(f"  Model Agreement: {prediction_result.get('model_agreement', 0):.2%}")
            
            print(f"\n💡 Recommended Strategy:")
            print(f"  Action: {strategy_result.get('action', 'N/A')}")
            print(f"  Expected Impact: {strategy_result.get('expected_impact', 'N/A')}")
            print(f"  Confidence: {strategy_result.get('confidence', 0):.2%}")
            
            print(f"\n⚡ Execution:")
            print(f"  Status: {execution_result.get('status', 'N/A')}")
            print(f"  Simulated Outcome: {execution_result.get('outcome', 'N/A')}")
            
            # Guardrail results
            guardrails = result.get('guardrails', {})
            print(f"\n🛡️  Guardrail Results:")
            print(f"  Layer 1 (Input): {'✓ Passed' if guardrails.get('layer1_passed') else '✗ Failed'}")
            print(f"  Layer 2 (Prediction): {'✓ Passed' if guardrails.get('layer2_passed') else '✗ Failed'}")
            print(f"  Layer 3 (Action): {'✓ Passed' if guardrails.get('layer3_passed') else '✗ Failed'}")
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Player processing failed: {error_msg}")
            print(f"\n❌ Processing failed: {error_msg}")
    
    # Monitor system health
    print(f"\n\n{'=' * 80}")
    print("🔍 SYSTEM HEALTH MONITORING")
    print(f"{'=' * 80}")
    
    health_report = orchestrator.monitor_system_health(
        X_val=X_test[:1000].values,
        y_val=y_test[:1000].values,
        X_baseline=X_train[:1000].values
    )
    
    if health_report.get('success'):
        drift_status = health_report.get('drift_detected', False)
        print(f"\n📊 Drift Detection: {'⚠️  DRIFT DETECTED' if drift_status else '✓ No drift detected'}")
        
        performance = health_report.get('performance', {})
        print(f"\n📈 Current Performance:")
        print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
        print(f"  Weighted F1: {performance.get('weighted_f1', 0):.3f}")
        
        if health_report.get('recommendation'):
            print(f"\n💭 Recommendation: {health_report['recommendation']}")
    
    # Display pipeline stats
    print(f"\n\n{'=' * 80}")
    print("📊 PIPELINE STATISTICS")
    print(f"{'=' * 80}")
    
    stats = orchestrator.get_pipeline_stats()
    print(f"\nAgent Executions:")
    for agent, count in stats.get('agent_executions', {}).items():
        print(f"  {agent}: {count}")
    
    print(f"\nGuardrail Performance:")
    guardrail_stats = stats.get('guardrail_stats', {})
    for layer, layer_stats in guardrail_stats.items():
        print(f"  {layer}: {layer_stats.get('passed', 0)}/{layer_stats.get('total', 0)} passed")
    
    print(f"\n{'=' * 80}")
    print("✅ DEMO COMPLETE")
    print(f"{'=' * 80}\n")


def run_interactive():
    """Run interactive mode for testing"""
    print("\n🎮 Interactive Agentic Gaming Analytics")
    print("=" * 80)
    print("\nCommands:")
    print("  'load' - Load dataset")
    print("  'train' - Train models")
    print("  'predict <player_id>' - Make prediction for a player")
    print("  'stats' - Show system statistics")
    print("  'health' - Check system health")
    print("  'quit' - Exit")
    print()
    
    orchestrator = AgenticOrchestrator()
    data = None
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                print("Goodbye!")
                break
            
            elif command == 'load':
                data_loader = DataLoader()
                data = data_loader.load_gaming_dataset()
                print(f"✓ Loaded {len(data)} player records")
            
            elif command == 'train':
                if data is None:
                    print("❌ Please load data first ('load' command)")
                    continue
                
                from sklearn.model_selection import train_test_split
                from src.utils.feature_engineering import FeatureEngineer
                
                feature_engineer = FeatureEngineer()
                X, y = feature_engineer.prepare_features(data)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                print("Training models...")
                result = orchestrator.train_pipeline(X_train, y_train)
                print("✓ Training complete!")
                
                if 'metrics' in result:
                    print(f"Accuracy: {result['metrics'].get('accuracy', 0):.3f}")
            
            elif command.startswith('predict'):
                if data is None:
                    print("❌ Please load data first")
                    continue
                
                parts = command.split()
                if len(parts) < 2:
                    print("Usage: predict <player_id>")
                    continue
                
                player_id = int(parts[1])
                # Find and predict for player
                print(f"Predicting for player {player_id}...")
                
            elif command == 'stats':
                stats = orchestrator.get_pipeline_stats()
                print("\n📊 System Statistics:")
                print(f"Agent executions: {stats.get('agent_executions', {})}")
                
            elif command == 'health':
                print("Checking system health...")
                # Would need validation data
                print("(Health check requires trained model and validation data)")
            
            else:
                print(f"Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic Gaming Analytics - Multi-Agent AI System"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive', 'train', 'eval'],
        default='demo',
        help='Run mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run appropriate mode
    if args.mode == 'demo':
        run_full_demo()
    elif args.mode == 'interactive':
        run_interactive()
    elif args.mode == 'train':
        print("Training mode not yet implemented")
    elif args.mode == 'eval':
        print("Evaluation mode not yet implemented")


if __name__ == "__main__":
    main()
