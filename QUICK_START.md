# Quick Start - Genetic Mutation Prioritization System

## 🚀 Launch the Web App (Fastest Method)

### Option 1: One Command Launch
```bash
python main.py --mode web
```

### Option 2: Simple Run Script  
```bash
python run_simple.py web
```

### Option 3: Development Mode
```bash
python main.py --mode web --debug
```

## 📋 What You Can Do Right Away

1. **🌐 Web Interface**: Drag & drop CSV files or enter mutations manually
2. **🎯 Model Selection**: Choose between MLP, Baseline, or Ensemble models  
3. **📊 Real-time Results**: Get instant predictions with confidence scores
4. **📈 Performance Metrics**: View system statistics and model performance

## 📁 Expected Input Format

Your CSV should have feature columns (optional columns like `mutation_id`, `gene`, `label` will be ignored):

```csv
feature_1,feature_2,feature_3,feature_4,...
0.5,0.3,0.8,0.2,...
0.7,0.4,0.6,0.9,...
```

## 🔧 Need Models?

If you see "No models found", you need to train models first:

```bash
# Start research environment
python main.py --mode research

# Then run these notebooks in order:
# 1. 00_data_pipeline.ipynb
# 2. 02_baseline_training.ipynb OR 03_mlp_training.ipynb
```

## 🛠️ Troubleshooting

**"No module named 'src'"?**  
→ Make sure you're in the project root directory

**"Port already in use"?**  
→ Use custom port: `python main.py --mode web --port 8080`

**Import errors?**  
→ Install requirements: `pip install -r requirements.txt`

**No models found?**  
→ Train models first using the research notebooks

## 🌐 Access the App

- **Local**: http://localhost:5000
- **Custom Port**: http://localhost:8080 (if using `--port 8080`)
- **Production**: http://your-server-ip:5000

For detailed documentation, see [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)