import plotly.graph_objects as go
import pandas as pd

def test_graph_logic():
    print("🔬 VERIFYING GRAPHING ENGINE...")
    
    # Simulated data matching our environment outputs
    history = [
        {"step": 1, "reward": 0.35, "logic": 0.85},
        {"step": 2, "reward": 0.72, "logic": 0.92},
        {"step": 3, "reward": 1.15, "logic": 0.88},
        {"step": 4, "reward": 1.50, "logic": 0.95},
    ]
    
    try:
        fig = go.Figure()
        steps = [h["step"] for h in history]
        
        # Reward Trace
        fig.add_trace(go.Scatter(
            x=steps, y=[h["reward"] for h in history], 
            name="Total Reward", 
            line=dict(color='#3b82f6', width=4)
        ))
        
        # Logic Trace
        fig.add_trace(go.Scatter(
            x=steps, y=[h["logic"] for h in history], 
            name="Logic Alignment", 
            line=dict(color='#7c3aed', dash='dot')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Sovereign Performance Trace (v5.5.0)",
            xaxis_title="Step",
            yaxis_title="Normalized Score"
        )
        
        print("✅ GRAPH LOGIC VERIFIED: No attribute errors detected.")
        return True
    except Exception as e:
        print(f"❌ GRAPH ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_graph_logic()
