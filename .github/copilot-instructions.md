# Copilot Instructions: PyTorch Learning Project

## Workspace Overview
This workspace contains the **mltest** project: educational PyTorch experiments for learning neural network fundamentals.

---

## Project: ML Test (PyTorch Learning)

### Overview
Educational PyTorch script ([test1.py](../test1.py)) demonstrating fundamental neural network training: data creation, model definition, loss computation, and backpropagation. Single file, ~100 lines, designed for learning PyTorch mechanics.

### Architecture & Data Flow
1. **Fake Data Generation** (lines 13–22): Random tensors (10 samples × 3 features) and targets (10 × 1)
2. **Model Definition** (lines 34–40): Sequential network: `3 input → 8 hidden (ReLU) → 8 hidden (ReLU) → 1 output`
3. **Loss Function** (line 47): Mean Squared Error (MSELoss)
4. **Optimizer** (line 54): Stochastic Gradient Descent (SGD) with learning rate 0.01
5. **Training Loop** (lines 63–89): 100 iterations of forward pass → loss → zero_grad → backward → weight update

### Key Code Patterns
- **Explicit Tensor Shapes**: All tensors documented with `.shape` comments (e.g., `torch.Size([10, 3])`)
- **Step-by-step Annotations**: Each training loop phase labeled (forward pass, loss, gradients, update)
- **Progress Monitoring**: Loss printed every 10 steps using `.item()` for scalar extraction
- **Linear Flow**: Sequential organization mirrors typical PyTorch workflow

### Current Status
✅ **Syntax correct** – `nn.Sequential()` properly constructed with all layers  
⚠️ **Educational scope** – All code in single file for learning clarity

### Known Limitations & Improvement Opportunities
- **No train/test split**: Model trains and evaluates on same data
- **No checkpointing**: Add `torch.save(model.state_dict(), 'model.pth')` after training to persist weights
- **Hardcoded hyperparameters**: Learning rate (0.01) and iterations (100) not configurable via CLI
- **Verbose output**: Loss printed every iteration (lines 70–72); consider sampling frequency for production
- **No validation loop**: Only training loss tracked; add separate validation phase for real projects

---

## Development Workflow

### Running the Script
```bash
python test1.py
```

**Expected Output:**
- Tensor shapes confirmed at startup (input: `[10, 3]`, target: `[10, 1]`)
- Loss values printed every 10 steps; values should generally decrease if training converges

### Common Modifications
1. **Change network architecture**: Edit `nn.Sequential()` block (lines 34–40)
   - Add layers: `nn.Linear(8, 4), nn.ReLU(),`
   - Adjust hidden units: change `nn.Linear(8, 8)` to larger/smaller values

2. **Adjust training dynamics**:
   - Learning rate: Line 54, `lr=0.01` (higher = faster but less stable)
   - Iterations: Line 63, `for i in range(100)` (more steps = better convergence, slower training)

3. **Inspect model internals**:
   - View layer 0 weights: `print(model[0].weight)`
   - Check gradients during training: `print(model[0].weight.grad)` inside loop

4. **Save/load trained model**:
   ```python
   # After training loop
   torch.save(model.state_dict(), 'model.pth')
   
   # To load later
   model.load_state_dict(torch.load('model.pth'))
   ```

---

## Code Organization Philosophy
- **Explicit over implicit**: All steps annotated; comments explain PyTorch mechanics
- **Shape-driven development**: Tensor shapes printed at key points to catch mismatches early
- **Linear flow**: Sequential structure (data → model → loss → training) mirrors typical PyTorch workflow
- **Debugging-friendly**: Print statements at each training step enable real-time monitoring

### Python Environment Requirements
- **PyTorch**: Required (`import torch`, `torch.nn`)
- **Python 3.7+**: Standard library only (no other dependencies)

### Next Steps for Enhancement
- [ ] Add train/test data split via `torch.utils.data.random_split()`
- [ ] Implement model checkpointing to save best weights
- [ ] Add command-line argument parsing for hyperparameters
- [ ] Log metrics to file or tensorboard for analysis
- [ ] Add validation loop separate from training
