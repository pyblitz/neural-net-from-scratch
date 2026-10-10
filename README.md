# Neural Network from Scratch 🧠

A tiny neural network (a **single neuron**) built in **pure Python** — no TensorFlow, no PyTorch, no ML libraries. Just math.

It **learns** the AND logic gate all by itself:

| Inputs | Target | Prediction |
|--------|--------|------------|
| 0, 0   | 0      | 0 ✅ |
| 0, 1   | 0      | 0 ✅ |
| 1, 0   | 0      | 0 ✅ |
| 1, 1   | 1      | 1 ✅ |

## How it works (step by step)

1. **Data** — 4 examples of the AND gate the network must learn
2. **Weights & bias** — start as random numbers
3. **Sigmoid function** — squashes any number into the 0–1 range
4. **Predict** — `sigmoid(weights × inputs + bias)`
5. **Training loop** — 1000 epochs. Each round it predicts, measures the **error**, and nudges the weights/bias to shrink it. *This is the learning!*
6. **Output** — all 4 predictions correct 🎉

## How to run

```bash
python neural_net.py
```

Needs only **Python 3** — no installs. You'll see the training finish and all 4 correct predictions printed.

## Try it yourself

- Change `random.seed(7)` to another number — it still learns!
- Change the data to the **OR gate** `([0,0],0), ([0,1],1), ([1,0],1), ([1,1],1)` — can it learn that too?

---

🎬 From the PyBlitz video **"I Built an AI in 40 Lines — No Libraries"**
📺 More Python projects: https://www.youtube.com/@codewithme9
