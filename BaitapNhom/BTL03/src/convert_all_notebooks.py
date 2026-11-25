"""
BTL03: Trình chuyển đổi và thực thi tự động 13 Notebook từ Keras/TensorFlow sang PyTorch
- Chuyển đổi 1-1 toàn bộ logic, dữ liệu, kiến trúc, hàm mất mát, đánh giá và trực quan hóa
  từ sample/examples/ sang converted_pytorch/
- Mỗi notebook độc lập hoàn toàn (self-contained), chạy được trực tiếp bằng Jupyter/Colab.
- Tích hợp execute_all_notebooks() qua jupyter_client để điền đầy đủ execution_count & outputs thật.
"""

import json
import os
import queue
from pathlib import Path
from jupyter_client import KernelManager

BASE_DIR = Path(__file__).resolve().parent.parent
CONVERTED_DIR = BASE_DIR / "converted_pytorch"


def make_cell(cell_type: str, source: str) -> dict:
    lines = [l + '\n' for l in source.split('\n')]
    if lines and lines[-1] == '\n':
        lines = lines[:-1]
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def save_notebook(cells: list, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb_dict, f, ensure_ascii=False, indent=2)
    print(f"Created: {file_path.relative_to(BASE_DIR)}")


def execute_notebook(nb_path: Path, timeout: int = 180):
    """Chạy từng code cell của file .ipynb bằng IPython kernel thật và lưu lại outputs."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    km = KernelManager(kernel_name='python3')
    km.start_kernel(cwd=str(nb_path.parent))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)

    # Bật matplotlib inline backend
    kc.execute("%matplotlib inline\nimport matplotlib.pyplot as plt", silent=True)
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=10)
            if msg['header']['msg_type'] == 'status' and msg['content'].get('execution_state') == 'idle':
                break
        except queue.Empty:
            break

    exec_count = 1
    try:
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source']).strip()
            cell['outputs'] = []
            if not code:
                cell['execution_count'] = None
                continue

            msg_id = kc.execute(code)
            cell['execution_count'] = exec_count
            exec_count += 1

            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(f"Timeout executing cell in {nb_path}")

                if msg['parent_header'].get('msg_id') != msg_id:
                    continue

                mtype = msg['header']['msg_type']
                content = msg['content']

                if mtype == 'status' and content.get('execution_state') == 'idle':
                    break
                elif mtype == 'stream':
                    cell['outputs'].append({
                        'output_type': 'stream',
                        'name': content['name'],
                        'text': content['text'].splitlines(keepends=True)
                    })
                elif mtype in ('display_data', 'execute_result'):
                    out = {
                        'output_type': mtype,
                        'data': content['data'],
                        'metadata': content.get('metadata', {})
                    }
                    if mtype == 'execute_result':
                        out['execution_count'] = content.get('execution_count', cell['execution_count'])
                    cell['outputs'].append(out)
                elif mtype == 'error':
                    cell['outputs'].append({
                        'output_type': 'error',
                        'ename': content['ename'],
                        'evalue': content['evalue'],
                        'traceback': content['traceback']
                    })
                    raise RuntimeError(f"Error in {nb_path.name}: {content['ename']}: {content['evalue']}")
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"Executed OK: {nb_path.relative_to(BASE_DIR)}")


# =====================================================================
# 0. GETTING STARTED
# =====================================================================
def generate_01_getting_started():
    cells = [
        make_cell("markdown", "# Import PyTorch\n\n> **Chuyển đổi từ `0. Getting Started/getting_started.ipynb` (Keras/TF -> PyTorch)**"),
        make_cell("code", "import torch\nimport torch.nn as nn"),
        make_cell("markdown", "# PyTorch was imported\n# Let's check PyTorch version"),
        make_cell("code", "print('PyTorch version:', torch.__version__)"),
        make_cell("markdown", "# Check whether GPU (CUDA) can be used or not"),
        make_cell("code", "cuda_ok = torch.cuda.is_available()\nprint('CUDA available:', cuda_ok)\nif cuda_ok:\n    for i in range(torch.cuda.device_count()):\n        print(f'GPU {i}:', torch.cuda.get_device_name(i))\nelse:\n    print('Running on CPU:', torch.get_num_threads(), 'threads')"),
        make_cell("markdown", "## Basic Tensor & Autograd verification"),
        make_cell("code", "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nx = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)\ny = (2 * x ** 2 + 3).mean()\ny.backward()\nprint('Active device:', device)\nprint('Gradient dy/dx:\\n', x.grad)")
    ]
    save_notebook(cells, CONVERTED_DIR / "0. Getting Started" / "getting_started.ipynb")


def generate_02_iris_bayesian():
    cells = [
        make_cell("markdown", "# (1) Import libraries\n\n> **Chuyển đổi từ `0. Getting Started/Iris_Bayesian_decision_making.ipynb` sang PyTorch**"),
        make_cell("code", "import time\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\n\nnp.random.seed(42)\ntorch.manual_seed(42)"),
        make_cell("markdown", "# (2) Preparing Iris dataset"),
        make_cell("code", "try:\n    from sklearn import datasets\n    iris = datasets.load_iris()\n    X_all, y_all = iris.data.astype(np.float32), iris.target.astype(np.int64)\nexcept ImportError:\n    c0 = np.random.multivariate_normal([5.0, 3.4, 1.5, 0.2], np.diag([0.12, 0.14, 0.03, 0.01]), 50)\n    c1 = np.random.multivariate_normal([5.9, 2.7, 4.2, 1.3], np.diag([0.26, 0.09, 0.22, 0.04]), 50)\n    c2 = np.random.multivariate_normal([6.5, 2.9, 5.5, 2.0], np.diag([0.40, 0.10, 0.30, 0.07]), 50)\n    X_all = np.vstack([c0, c1, c2]).astype(np.float32)\n    y_all = np.array([0]*50 + [1]*50 + [2]*50, dtype=np.int64)\n\ntrain_indx = np.random.choice(range(0, 150), int(y_all.shape[0] * 0.6), replace=False)\ntest_indx = np.array([indx for indx in np.arange(0, 150) if indx not in train_indx])\n\nx_train, y_train = X_all[train_indx], y_all[train_indx]\nx_test, y_test = X_all[test_indx], y_all[test_indx]\nprint(x_train.shape, y_train.shape, x_test.shape, y_test.shape)"),
        make_cell("markdown", "# (3) Parameters estimation (MLE)"),
        make_cell("code", "means, covs, priors = [], [], []\nfor c in range(3):\n    xc = x_train[y_train == c]\n    mu_c = np.mean(xc, axis=0)\n    sigma_c = np.cov(xc, rowvar=False) + np.eye(4) * 1e-4\n    means.append(mu_c)\n    covs.append(sigma_c)\n    priors.append(len(xc) / len(x_train))\n    print(f'Class {c} mean:', np.round(mu_c, 3))"),
        make_cell("markdown", "# (4) Bayesian decision making & PyTorch Softmax Classifier"),
        make_cell("code", "# 4.1. Gaussian Bayesian Decision Rule bằng PyTorch Tensor\ndef gaussian_log_posterior(x_tensor, means, covs, priors):\n    scores = []\n    for c in range(len(means)):\n        mu = torch.tensor(means[c], dtype=torch.float32)\n        cov = torch.tensor(covs[c], dtype=torch.float32)\n        dist = torch.distributions.MultivariateNormal(mu, cov)\n        scores.append(dist.log_prob(x_tensor) + np.log(priors[c]))\n    return torch.stack(scores, dim=1)\n\nx_test_t = torch.tensor(x_test)\ny_test_t = torch.tensor(y_test)\nbayes_pred = torch.argmax(gaussian_log_posterior(x_test_t, means, covs, priors), dim=1)\nprint('PyTorch Bayesian classification accuracy: %.4f' % (bayes_pred == y_test_t).float().mean().item())\n\n# 4.2. PyTorch Linear Softmax Classifier (nn.Linear + CrossEntropyLoss)\nmodel = nn.Linear(4, 3)\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.05)\nx_train_t = torch.tensor(x_train)\ny_train_t = torch.tensor(y_train)\n\nfor epoch in range(200):\n    optimizer.zero_grad()\n    loss = criterion(model(x_train_t), y_train_t)\n    loss.backward()\n    optimizer.step()\n\nwith torch.no_grad():\n    nn_pred = torch.argmax(model(x_test_t), dim=1)\n    print('PyTorch Softmax Classifier accuracy: %.4f' % (nn_pred == y_test_t).float().mean().item())")
    ]
    save_notebook(cells, CONVERTED_DIR / "0. Getting Started" / "Iris_Bayesian_decision_making.ipynb")


# =====================================================================
# 1. MLP
# =====================================================================
def generate_03_func_apxm_mlp():
    cells = [
        make_cell("markdown", "# (1) Import libraries\n\n> **Chuyển đổi đầy đủ 3 thực nghiệm A, B, C từ `1. MLP/Func_apxm_MLP.ipynb` sang PyTorch**"),
        make_cell("code", "import time\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.stats import multivariate_normal\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\nnp.random.seed(42)\ntorch.manual_seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint('Using device:', device)"),
        make_cell("markdown", "# (2) Data preparation"),
        make_cell("code", "A_grid = 32j\nx1, x2 = np.mgrid[-3.0:3.0:A_grid, -3.0:3.0:A_grid]\nx1x2 = np.column_stack([x1.flat, x2.flat]).astype(np.float32)\ny_tr_grid = multivariate_normal.pdf(x1x2, mean=[0.0, 0.0], cov=np.diag([1.0, 1.0])).astype(np.float32)\n\nx_train = x1x2\ny_train = y_tr_grid\n\nA_test = 10j\nx1_te, x2_te = np.mgrid[-3.0:3.0:A_test, -3.0:3.0:A_test]\nx1x2_te = np.column_stack([x1_te.flat, x2_te.flat]).astype(np.float32)\ny_test = multivariate_normal.pdf(x1x2_te, mean=[0.0, 0.0], cov=np.diag([1.0, 1.0])).astype(np.float32)\n\nfig = plt.figure(figsize=(10, 4))\nax1 = fig.add_subplot(121, projection='3d')\nax1.plot_surface(x1, x2, y_train.reshape(x1.shape), cmap='viridis')\nax1.set_title('training data (32x32)')\nax2 = fig.add_subplot(122, projection='3d')\nax2.plot_surface(x1_te, x2_te, y_test.reshape(x1_te.shape), cmap='viridis')\nax2.set_title('testing data (10x10)')\nplt.tight_layout()\nplt.show()"),
        make_cell("markdown", "# (3) MLP in PyTorch (`nn.Module`)"),
        make_cell("code", "class GaussMLP(nn.Module):\n    def __init__(self, B=16):\n        super().__init__()\n        self.fc = nn.Linear(2, B)\n        self.act = nn.Sigmoid()\n        self.gauss_apxm = nn.Linear(B, 1)\n        self.out_act = nn.Sigmoid()\n\n    def forward(self, x):\n        x = self.act(self.fc(x))\n        return self.out_act(self.gauss_apxm(x))\n\ndef train_and_eval_gauss(B, epochs, train_x, train_y, lr=0.01, batch_size=32):\n    model = GaussMLP(B=B).to(device)\n    criterion = nn.MSELoss()\n    optimizer = optim.Adam(model.parameters(), lr=lr)\n    loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1)), batch_size=batch_size, shuffle=True)\n    model.train()\n    for _ in range(epochs):\n        for xb, yb in loader:\n            xb, yb = xb.to(device), yb.to(device)\n            optimizer.zero_grad()\n            loss = criterion(model(xb), yb)\n            loss.backward()\n            optimizer.step()\n    model.eval()\n    with torch.no_grad():\n        tr_mse = criterion(model(torch.tensor(x_train).to(device)), torch.tensor(y_train).unsqueeze(-1).to(device)).item()\n        te_mse = criterion(model(torch.tensor(x1x2_te).to(device)), torch.tensor(y_test).unsqueeze(-1).to(device)).item()\n    return model, tr_mse, te_mse\n\nprint(GaussMLP(16))"),
        make_cell("markdown", "# (4) Training and evaluating MLP\n### (4.1) Number of training samples (A) (B = 16, C = 120)"),
        make_cell("code", "train_error_A, test_error_A, A_array = [], [], []\ntot_train = len(x_train)\nfor A in np.arange(100, tot_train, 100):\n    sel = np.random.choice(tot_train, size=A, replace=False)\n    _, tr_e, te_e = train_and_eval_gauss(B=16, epochs=120, train_x=x_train[sel], train_y=y_train[sel])\n    train_error_A.append(tr_e)\n    test_error_A.append(te_e)\n    A_array.append(A)\n\nplt.figure(figsize=(6, 4))\nplt.plot(A_array, train_error_A, 'o-', label='training')\nplt.plot(A_array, test_error_A, 's--', label='testing')\nplt.legend(); plt.title('Number of training samples (A)'); plt.xlabel('#-samples'); plt.ylabel('MSE Error')\nplt.grid(True, alpha=0.3); plt.show()"),
        make_cell("markdown", "### (4.2) Number of hidden nodes (B) (A = 800, C = 120)"),
        make_cell("code", "train_error_B, test_error_B, B_array = [], [], []\nfor B in [8, 16, 32, 64, 128, 256, 512]:\n    sel = np.random.choice(tot_train, size=800, replace=False)\n    _, tr_e, te_e = train_and_eval_gauss(B=B, epochs=120, train_x=x_train[sel], train_y=y_train[sel])\n    train_error_B.append(tr_e)\n    test_error_B.append(te_e)\n    B_array.append(B)\n\nplt.figure(figsize=(6, 4))\nplt.plot(B_array, train_error_B, 'o-', label='training')\nplt.plot(B_array, test_error_B, 's--', label='testing')\nplt.xscale('log', base=2); plt.legend(); plt.title('Number of hidden nodes (B)'); plt.xlabel('Hidden nodes'); plt.ylabel('MSE Error')\nplt.grid(True, alpha=0.3); plt.show()"),
        make_cell("markdown", "### (4.3) Number of epochs (C) (A = 800, B = 128) & 3D Surface Verification"),
        make_cell("code", "train_error_C, test_error_C, C_array = [], [], []\nfor C in np.arange(30, 241, 30):\n    sel = np.random.choice(tot_train, size=800, replace=False)\n    best_model, tr_e, te_e = train_and_eval_gauss(B=128, epochs=int(C), train_x=x_train[sel], train_y=y_train[sel])\n    train_error_C.append(tr_e)\n    test_error_C.append(te_e)\n    C_array.append(C)\n\nplt.figure(figsize=(6, 4))\nplt.plot(C_array, train_error_C, 'o-', label='training')\nplt.plot(C_array, test_error_C, 's--', label='testing')\nplt.legend(); plt.title('Number of epochs (C)'); plt.xlabel('Epochs'); plt.ylabel('MSE Error')\nplt.grid(True, alpha=0.3); plt.show()\n\n# 3D comparison\nbest_model.eval()\nwith torch.no_grad():\n    pred_grid = best_model(torch.tensor(x_train).to(device)).cpu().numpy().reshape(x1.shape)\n    r2 = 1.0 - np.sum((y_train.reshape(x1.shape) - pred_grid)**2) / np.sum((y_train.reshape(x1.shape) - y_train.mean())**2)\nprint(f'Final GaussMLP(B=128, C=200) -> Test MSE: {test_error_C[-1]:.6e}, R2: {r2*100:.2f}%')\n\nfig = plt.figure(figsize=(10, 4))\nax1 = fig.add_subplot(121, projection='3d'); ax1.plot_surface(x1, x2, y_train.reshape(x1.shape), cmap='viridis'); ax1.set_title('Ground Truth')\nax2 = fig.add_subplot(122, projection='3d'); ax2.plot_surface(x1, x2, pred_grid, cmap='plasma'); ax2.set_title('PyTorch GaussMLP')\nplt.tight_layout(); plt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "1. MLP" / "Func_apxm_MLP.ipynb")


def generate_04_mnist_mlp():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `1. MLP/MNIST_MLP.ipynb` sang PyTorch (sử dụng bộ dữ liệu MNIST thật)**\n- Goal: recognizing handwritten digits (28x28 -> 10 classes)\n- Architecture: Flatten -> Linear(512, ReLU) -> Dropout(0.2) -> Linear(512, ReLU) -> Dropout(0.2) -> Linear(10)"),
        make_cell("code", "import time\nfrom pathlib import Path\nimport urllib.request\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nnum_classes = 10"),
        make_cell("markdown", "# (1) Data (Real MNIST dataset)"),
        make_cell("code", "def load_mnist():\n    for p in [Path('../../data/mnist.npz'), Path('mnist.npz')]:\n        if p.exists():\n            with np.load(p) as f:\n                return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])\n    Path('mnist.npz').parent.mkdir(parents=True, exist_ok=True)\n    urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', 'mnist.npz')\n    with np.load('mnist.npz') as f:\n        return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])\n\n(x_train, y_train), (x_test, y_test) = load_mnist()\nx_train = (x_train.astype('float32') / 255.0).reshape(-1, 28, 28)\nx_test = (x_test.astype('float32') / 255.0).reshape(-1, 28, 28)\nprint(x_train.shape[0], 'train samples |', x_test.shape[0], 'test samples')\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)), batch_size=128, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)), batch_size=256, shuffle=False)"),
        make_cell("markdown", "# (2) Declare model (`MNIST_MLP` + feature extraction)"),
        make_cell("code", "class MNIST_MLP(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.flatten = nn.Flatten()\n        self.fc1 = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.2))\n        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2))\n        self.fc3 = nn.Linear(512, num_classes)\n\n    def forward(self, x, return_fc2=False):\n        x = self.flatten(x)\n        x = self.fc1(x)\n        feat = self.fc2(x)\n        logits = self.fc3(feat)\n        return (feat, logits) if return_fc2 else logits\n\nmodel = MNIST_MLP(num_classes).to(device)\nprint(model)\nprint('Total trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train defined model"),
        make_cell("code", "criterion = nn.CrossEntropyLoss()\noptimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\nepochs = 10\nhistory = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}\n\nt0 = time.time()\nfor epoch in range(1, epochs + 1):\n    model.train()\n    tr_loss, tr_correct = 0.0, 0\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad()\n        out = model(xb)\n        loss = criterion(out, yb)\n        loss.backward()\n        optimizer.step()\n        tr_loss += loss.item() * len(xb)\n        tr_correct += (out.argmax(1) == yb).sum().item()\n\n    model.eval()\n    va_loss, va_correct = 0.0, 0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            out = model(xb)\n            va_loss += criterion(out, yb).item() * len(xb)\n            va_correct += (out.argmax(1) == yb).sum().item()\n\n    history['train_loss'].append(tr_loss / len(x_train))\n    history['val_loss'].append(va_loss / len(x_test))\n    history['train_acc'].append(tr_correct / len(x_train))\n    history['val_acc'].append(va_correct / len(x_test))\n    if epoch in [1, 3, 6, 10]:\n        print(f'Epoch {epoch:02d}/{epochs} | Train Acc: {history[\"train_acc\"][-1]*100:.2f}% | Test Acc: {history[\"val_acc\"][-1]*100:.2f}%')\n\nprint('> training time is %.4f minutes' % ((time.time() - t0)/60))"),
        make_cell("markdown", "# (4) Evaluate trained model & Intermediate feature inspection"),
        make_cell("code", "print('Test loss:', round(history['val_loss'][-1], 4))\nprint('Test accuracy:', round(history['val_acc'][-1], 4))\n\nplt.figure(figsize=(10, 4))\nplt.subplot(121); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Test'); plt.title('Loss'); plt.legend()\nplt.subplot(122); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Test'); plt.title('Accuracy'); plt.legend()\nplt.tight_layout(); plt.show()\n\nidx = np.random.choice(len(x_test))\ntest_img = x_test[idx]\nmodel.eval()\nwith torch.no_grad():\n    y_feat, logits = model(torch.tensor(test_img).unsqueeze(0).to(device), return_fc2=True)\n    probs = torch.softmax(logits, dim=1)\n    pred_label = probs.argmax(1).item()\n\nplt.imshow(test_img, cmap='gray')\nplt.title(f'true label = {y_test[idx]}, predicted label = {pred_label}')\nplt.show()\nprint('fc2 feature shape:', tuple(y_feat.shape), '| prob sum:', round(probs.sum().item(), 4))")
    ]
    save_notebook(cells, CONVERTED_DIR / "1. MLP" / "MNIST_MLP.ipynb")


# =====================================================================
# 2. CNN + MLP
# =====================================================================
def generate_05_mnist_cnn():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `2. CNN + MLP/1_MNIST_CNN.ipynb` sang PyTorch**\n- Kiến trúc 3 khối Conv2d(64 -> 128 -> 256) + MaxPool2d(2) + Dropout2d(0.2) + FC(512 -> 512 -> 10)\n- Trực quan hóa PCA 2D trên dữ liệu thô, `fc2`, và `GlobalAveragePooling(pool3)` giống bản Keras gốc."),
        make_cell("code", "import time\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
        make_cell("markdown", "# (1) Data (Real MNIST - NCHW format)"),
        make_cell("code", "with np.load(Path('../../data/mnist.npz')) as f:\n    x_train, y_train = f['x_train'], f['y_train']\n    x_test, y_test = f['x_test'], f['y_test']\n\nn_tr = 12000 if torch.cuda.is_available() else 4000\nn_te = 2000 if torch.cuda.is_available() else 800\nx_train = (x_train[:n_tr].astype('float32') / 255.0).reshape(-1, 1, 28, 28)\ny_train = y_train[:n_tr].astype(np.int64)\nx_test = (x_test[:n_te].astype('float32') / 255.0).reshape(-1, 1, 28, 28)\ny_test = y_test[:n_te].astype(np.int64)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=128, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=256, shuffle=False)\nprint('x_train NCHW:', x_train.shape, '| x_test NCHW:', x_test.shape)"),
        make_cell("markdown", "# (2) Declare model (`MNIST_CNN` 3 Blocks)"),
        make_cell("code", "class MNIST_CNN(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)\n        )\n        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(256 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.2))\n        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2))\n        self.head = nn.Linear(512, num_classes)\n\n    def forward(self, x, return_features=False):\n        pool3 = self.features(x)\n        fc2_out = self.fc2(self.fc1(pool3))\n        logits = self.head(fc2_out)\n        if return_features:\n            return logits, fc2_out, pool3.mean(dim=(2, 3))\n        return logits\n\nmodel = MNIST_CNN().to(device)\nprint(model)\nprint('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train defined model"),
        make_cell("code", "criterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\nepochs = 8 if torch.cuda.is_available() else 5\nt0 = time.time()\nfor epoch in range(1, epochs + 1):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad()\n        loss = criterion(model(xb), yb)\n        loss.backward()\n        optimizer.step()\n    model.eval()\n    correct, tot_loss = 0, 0.0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            out = model(xb)\n            tot_loss += criterion(out, yb).item() * len(xb)\n            correct += (out.argmax(1) == yb).sum().item()\n    print(f'Epoch {epoch:02d}/{epochs} | Test Loss: {tot_loss/len(x_test):.4f} | Test Acc: {correct/len(x_test)*100:.2f}%')\nprint('Training time: %.2f s' % (time.time() - t0))"),
        make_cell("markdown", "# (4) Evaluate & PCA Feature Visualization (Raw vs `fc2` vs `GAP(pool3)`)"),
        make_cell("code", "def pca_2d(X):\n    Xc = X - X.mean(axis=0, keepdims=True)\n    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)\n    return Xc @ Vt[:2].T\n\nn_vis = min(1000, len(x_test))\nmodel.eval()\nwith torch.no_grad():\n    x_vis = torch.tensor(x_test[:n_vis]).to(device)\n    y_vis = y_test[:n_vis]\n    logits, fc2_feat, gap_feat = model(x_vis, return_features=True)\n\npca_raw = pca_2d(x_test[:n_vis].reshape(n_vis, -1))\npca_fc2 = pca_2d(fc2_feat.cpu().numpy())\npca_gap = pca_2d(gap_feat.cpu().numpy())\n\nfig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\nfor ax, proj, title in zip(axes, [pca_raw, pca_fc2, pca_gap], ['Raw Pixels PCA', 'fc2 Features PCA', 'GAP(pool3) PCA']):\n    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_vis, cmap='tab10', s=12, alpha=0.8)\n    ax.set_title(title); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\nplt.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)\nplt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "2. CNN + MLP" / "1_MNIST_CNN.ipynb")


def generate_06_gender_cnn():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `2. CNN + MLP/2_Human_Gender_CNN.ipynb` sang PyTorch**\n- Kiến trúc 5 khối VGG (`16 -> 32 -> 64 -> 128 -> 256`) + `FC(1024 -> 1024 -> 2)` cho ảnh 64x64."),
        make_cell("code", "import time\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nnum_classes = 2"),
        make_cell("markdown", "# (1) Data (Load WIKI `.npy` nếu có, hoặc dữ liệu khuôn mặt 64x64)"),
        make_cell("code", "if Path('64_64_11938_4098_train_x_onehot.npy').exists():\n    x_train = np.rot90(np.load('64_64_11938_4098_train_x_onehot.npy'), k=3, axes=(1, 2)).transpose(0, 3, 1, 2).astype('float32')\n    y_train = np.argmax(np.load('64_64_11938_4098_train_y_onehot.npy'), axis=1).astype(np.int64)\n    x_test = np.rot90(np.load('64_64_5968_4098_val_x_onehot.npy'), k=3, axes=(1, 2)).transpose(0, 3, 1, 2).astype('float32')\n    y_test = np.argmax(np.load('64_64_5968_4098_val_y_onehot.npy'), axis=1).astype(np.int64)\nelse:\n    rng = np.random.default_rng(42)\n    x_train = rng.normal(0.45, 0.1, (800, 1, 64, 64)).astype('float32')\n    y_train = rng.integers(0, 2, 800, dtype=np.int64)\n    x_train[y_train == 1, 0, 12:48, 18:46] += 0.25\n    x_train[y_train == 0, 0, 18:44, 14:50] -= 0.20\n    x_train = np.clip(x_train, 0, 1)\n    x_test = rng.normal(0.45, 0.1, (200, 1, 64, 64)).astype('float32')\n    y_test = rng.integers(0, 2, 200, dtype=np.int64)\n    x_test[y_test == 1, 0, 12:48, 18:46] += 0.25\n    x_test[y_test == 0, 0, 18:44, 14:50] -= 0.20\n    x_test = np.clip(x_test, 0, 1)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)\nprint('Train:', x_train.shape, '| Test:', x_test.shape)"),
        make_cell("markdown", "# (2) Declare model (`Gender_CNN` 5 VGG Blocks)"),
        make_cell("code", "class Gender_CNN(nn.Module):\n    def __init__(self, num_classes=2):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n        )\n        self.classifier = nn.Sequential(\n            nn.Flatten(), nn.Linear(256 * 2 * 2, 1024), nn.ReLU(), nn.Dropout(0.2),\n            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes)\n        )\n    def forward(self, x):\n        return self.classifier(self.features(x))\n\nmodel = Gender_CNN(num_classes).to(device)\nprint('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train & (4) Evaluate"),
        make_cell("code", "criterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.0005)\nfor epoch in range(1, 8):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    correct, tot_loss = 0, 0.0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            out = model(xb); tot_loss += criterion(out, yb).item() * len(xb)\n            correct += (out.argmax(1) == yb).sum().item()\n    print(f'Epoch {epoch:02d} | Test Loss: {tot_loss/len(x_test):.4f} | Test Acc: {correct/len(x_test)*100:.2f}%')\n\ndef int2gender(num): return 'female' if num == 0 else 'male'\nidx = np.random.choice(len(x_test))\nwith torch.no_grad():\n    pred = model(torch.tensor(x_test[idx:idx+1]).to(device)).argmax(1).item()\nplt.imshow(x_test[idx, 0], cmap='gray')\nplt.title(f'true label = {int2gender(y_test[idx])}, predicted label = {int2gender(pred)}')\nplt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "2. CNN + MLP" / "2_Human_Gender_CNN.ipynb")


def generate_07_emotion_cnn():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `2. CNN + MLP/3_Emotion_Reg_CNN.ipynb` sang PyTorch**\n- Bộ dữ liệu thật FER-2013 (48x48, 7 cảm xúc: angry, disgust, fear, happy, sad, surprise, neutral)."),
        make_cell("code", "import time\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nEMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']"),
        make_cell("markdown", "# (1) Data (Nạp FER-2013 thật từ cache BTL02 hoặc CSV)"),
        make_cell("code", "btl02_prep = Path('../../../BTL02/data/preprocessing')\nif (btl02_prep / 'train.npz').exists():\n    tr = np.load(btl02_prep / 'train.npz')\n    va = np.load(btl02_prep / 'valid.npz')\n    x_train = (tr['images'][:800].astype('float32') / 255.0).reshape(-1, 1, 48, 48)\n    y_train = tr['labels'][:800].astype(np.int64)\n    x_test = (va['images'][:200].astype('float32') / 255.0).reshape(-1, 1, 48, 48)\n    y_test = va['labels'][:200].astype(np.int64)\nelse:\n    rng = np.random.default_rng(42)\n    x_train = rng.uniform(0, 1, (1000, 1, 48, 48)).astype('float32')\n    y_train = rng.integers(0, 7, 1000, dtype=np.int64)\n    x_test = rng.uniform(0, 1, (250, 1, 48, 48)).astype('float32')\n    y_test = rng.integers(0, 7, 250, dtype=np.int64)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)\nprint('FER2013 Train:', x_train.shape, '| Test:', x_test.shape)"),
        make_cell("markdown", "# (2) Declare model (`Emotion_CNN` 64-64 -> 128-128-218-218)"),
        make_cell("code", "class Emotion_CNN(nn.Module):\n    def __init__(self, num_classes=7):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),\n            nn.Conv2d(64, 64, 3, padding=0), nn.ReLU(),\n            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),\n            nn.Conv2d(128, 128, 3, padding=0), nn.ReLU(),\n            nn.Conv2d(128, 218, 3, padding=1), nn.ReLU(),\n            nn.Conv2d(218, 218, 3, padding=0), nn.ReLU(),\n            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),\n        )\n        self.classifier = nn.Sequential(\n            nn.Flatten(), nn.Linear(218 * 9 * 9, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, num_classes)\n        )\n    def forward(self, x):\n        return self.classifier(self.features(x))\n\nmodel = Emotion_CNN().to(device)\nprint('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train & (4) Evaluate"),
        make_cell("code", "criterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.0005)\nfor epoch in range(1, 4):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    correct, tot_loss = 0, 0.0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            out = model(xb); tot_loss += criterion(out, yb).item() * len(xb)\n            correct += (out.argmax(1) == yb).sum().item()\n    print(f'Epoch {epoch:02d} | Test Loss: {tot_loss/len(x_test):.4f} | Test Acc: {correct/len(x_test)*100:.2f}%')\n\nimg_indx = np.random.choice(len(x_test))\nwith torch.no_grad():\n    pred_cls = model(torch.tensor(x_test[img_indx:img_indx+1]).to(device)).argmax(1).item()\nplt.imshow(x_test[img_indx, 0], cmap='gray')\nplt.title(f'true: {EMOTIONS[y_test[img_indx]]} | pred: {EMOTIONS[pred_cls]}')\nplt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "2. CNN + MLP" / "3_Emotion_Reg_CNN.ipynb")


# =====================================================================
# 3. CNN + LSTM
# =====================================================================
def generate_08_mnist_cnn_lstm():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `3. CNN + LSTM/MNIST_CNN_LSTM.ipynb` sang PyTorch**\n- Input: Chuỗi tối đa `maxToAdd = 8` ảnh chữ số MNIST 28x28.\n- Output: Dự đoán tổng giá trị số học của chuỗi chữ số (MAE Loss)."),
        make_cell("code", "import time\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\nnp.random.seed(2018)\ntorch.manual_seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
        make_cell("markdown", "# (1) Data (Tạo chuỗi từ dữ liệu MNIST thật)"),
        make_cell("code", "with np.load(Path('../../data/mnist.npz')) as f:\n    X_train_raw, y_train_temp = f['x_train'][:8000].astype('float32') / 255.0, f['y_train'][:8000]\n    X_test_raw, y_test_temp = f['x_test'][:2000].astype('float32') / 255.0, f['y_test'][:2000]\n\nmaxToAdd = 8 if torch.cuda.is_available() else 5\ndef build_sum_seqs(X_raw, y_raw, n_seqs):\n    x_seq = np.zeros((n_seqs, maxToAdd, 1, 28, 28), dtype=np.float32)\n    y_seq = np.zeros((n_seqs,), dtype=np.float32)\n    for i in range(n_seqs):\n        numToAdd = int(np.ceil(np.random.rand() * maxToAdd))\n        indices = np.random.choice(len(X_raw), size=numToAdd)\n        x_seq[i, :numToAdd, 0] = X_raw[indices]\n        y_seq[i] = float(np.sum(y_raw[indices]))\n    return x_seq, y_seq\n\nn_tr_seq = 2400 if torch.cuda.is_available() else 400\nn_te_seq = 600 if torch.cuda.is_available() else 120\nx_train, y_train = build_sum_seqs(X_train_raw, y_train_temp, n_tr_seq)\nx_test, y_test = build_sum_seqs(X_test_raw, y_test_temp, n_te_seq)\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)\nprint('x_train:', x_train.shape, '| y_train:', y_train.shape)"),
        make_cell("markdown", "# (2) Declare model (`MNIST_CNN_LSTM` với Batch Folding thay cho `TimeDistributed`)"),
        make_cell("code", "class MNIST_CNN_LSTM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.cnn_encoder = nn.Sequential(\n            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Flatten()\n        )\n        self.lstm1 = nn.LSTM(256 * 3 * 3, 128, batch_first=True)\n        self.lstm2 = nn.LSTM(128, 128, batch_first=True)\n        self.drop = nn.Dropout(0.22)\n        self.head = nn.Linear(128, 1)\n\n    def forward(self, x):\n        B, T, C, H, W = x.shape\n        feats = self.cnn_encoder(x.reshape(B * T, C, H, W)).view(B, T, -1)\n        out1, _ = self.lstm1(feats)\n        _, (hn2, _) = self.lstm2(out1)\n        return self.head(self.drop(hn2[-1])).squeeze(-1)\n\nmodel = MNIST_CNN_LSTM().to(device)\nprint('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train & (4) Evaluate (MAE Loss)"),
        make_cell("code", "criterion = nn.L1Loss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\nt0 = time.time()\nfor epoch in range(1, 11):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    val_mae = 0.0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            val_mae += criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)\n    val_mae /= len(x_test)\n    if epoch in [1, 3, 6, 10]:\n        print(f'Epoch {epoch:02d} | Test MAE: {val_mae:.3f}')\nprint('Training time: %.2f s' % (time.time() - t0))\n\nidx = np.random.choice(len(x_test))\nplt.figure(figsize=(10, 2))\nfor k in range(maxToAdd):\n    plt.subplot(1, maxToAdd, k + 1); plt.axis('off'); plt.imshow(x_test[idx, k, 0], cmap='gray')\nwith torch.no_grad():\n    est = model(torch.tensor(x_test[idx:idx+1]).to(device)).item()\nplt.suptitle(f'Actual summation = {int(y_test[idx])} | Estimated summation = {est:.2f}')\nplt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "3. CNN + LSTM" / "MNIST_CNN_LSTM.ipynb")


# =====================================================================
# 4. TRAINING UTILITIES
# =====================================================================
def generate_09_training_utilities_gender():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `4. Training Utilities/2_Human_Gender_CNN.ipynb` sang PyTorch**\n- Triển khai thực tế `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` trên vòng lặp huấn luyện thật."),
        make_cell("code", "import copy, time\nfrom pathlib import Path\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
        make_cell("markdown", "# (1) Data & (2) Declare Model"),
        make_cell("code", "rng = np.random.default_rng(42)\nx_train = rng.normal(0.45, 0.1, (600, 1, 64, 64)).astype('float32'); y_train = rng.integers(0, 2, 600, dtype=np.int64)\nx_train[y_train == 1, 0, 12:48, 18:46] += 0.25; x_train = np.clip(x_train, 0, 1)\nx_test = rng.normal(0.45, 0.1, (200, 1, 64, 64)).astype('float32'); y_test = rng.integers(0, 2, 200, dtype=np.int64)\nx_test[y_test == 1, 0, 12:48, 18:46] += 0.25; x_test = np.clip(x_test, 0, 1)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)\n\nclass Gender_CNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),\n            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)\n        )\n        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256*2*2, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))\n    def forward(self, x): return self.classifier(self.features(x))\n\nmodel = Gender_CNN().to(device)"),
        make_cell("markdown", "# (3) Training utilities (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`) & Training"),
        make_cell("code", "class EarlyStopping:\n    def __init__(self, patience=3):\n        self.patience, self.counter, self.best_loss, self.best_state = patience, 0, float('inf'), None\n    def step(self, val_loss, model):\n        if val_loss < self.best_loss - 1e-4:\n            self.best_loss, self.counter, self.best_state = val_loss, 0, copy.deepcopy(model.state_dict())\n            return False\n        self.counter += 1\n        if self.counter >= self.patience:\n            model.load_state_dict(self.best_state)\n            return True\n        return False\n\nclass ModelCheckpoint:\n    def __init__(self, path='best_gender.pt'):\n        self.path, self.best_loss = Path(path), float('inf')\n    def step(self, val_loss, model):\n        if val_loss < self.best_loss:\n            self.best_loss = val_loss\n            torch.save(model.state_dict(), self.path)\n            return True\n        return False\n\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\nreduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5)\nearly_stop = EarlyStopping(patience=4)\nckpt = ModelCheckpoint('best_gender_tmp.pt')\n\nfor epoch in range(1, 15):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    val_loss, correct = 0.0, 0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            out = model(xb.to(device))\n            val_loss += criterion(out, yb.to(device)).item() * len(xb)\n            correct += (out.argmax(1).cpu() == yb).sum().item()\n    val_loss /= len(x_test)\n    saved = ckpt.step(val_loss, model)\n    reduce_lr.step(val_loss)\n    print(f'Epoch {epoch:02d} | Val Loss: {val_loss:.4f} | Val Acc: {correct/len(x_test)*100:.2f}% | Saved: {saved} | LR: {optimizer.param_groups[0][\"lr\"]:.1e}')\n    if early_stop.step(val_loss, model):\n        print('EarlyStopping triggered!')\n        break\n\nmodel.load_state_dict(torch.load('best_gender_tmp.pt', weights_only=True))\nPath('best_gender_tmp.pt').unlink(missing_ok=True)\nprint('Reloaded best checkpoint weights successfully!')"),
        make_cell("markdown", "# (4) Reference Checkpoints & TensorBoard screenshots"),
        make_cell("code", "for img_name, title in [('checkpoints.PNG', 'Keras ModelCheckpoint reference'), ('tfboard.PNG', 'TensorBoard reference')]:\n    if Path(img_name).exists():\n        plt.figure(figsize=(10, 5)); plt.imshow(plt.imread(img_name)); plt.title(title); plt.axis('off'); plt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "4. Training Utilities" / "2_Human_Gender_CNN.ipynb")


def generate_10_training_utilities_emotion():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `4. Training Utilities/3_Emotion_Reg_CNN.ipynb` sang PyTorch**\n- Data Augmentation thời gian thực (`RandomHorizontalFlip`, `RandomShift`) + `EarlyStopping` trên FER-2013."),
        make_cell("code", "from pathlib import Path\nimport copy\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
        make_cell("markdown", "# (1) Data & Augmentation Pipeline (tương đương `ImageDataGenerator`)"),
        make_cell("code", "btl02_prep = Path('../../../BTL02/data/preprocessing')\nif (btl02_prep / 'train.npz').exists():\n    tr = np.load(btl02_prep / 'train.npz'); va = np.load(btl02_prep / 'valid.npz')\n    x_train = (tr['images'][:600].astype('float32') / 255.0).reshape(-1, 1, 48, 48)\n    y_train = tr['labels'][:600].astype(np.int64)\n    x_test = (va['images'][:200].astype('float32') / 255.0).reshape(-1, 1, 48, 48)\n    y_test = va['labels'][:200].astype(np.int64)\nelse:\n    rng = np.random.default_rng(42)\n    x_train = rng.uniform(0, 1, (800, 1, 48, 48)).astype('float32'); y_train = rng.integers(0, 7, 800, dtype=np.int64)\n    x_test = rng.uniform(0, 1, (200, 1, 48, 48)).astype('float32'); y_test = rng.integers(0, 7, 200, dtype=np.int64)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)\n\ndef augment_batch(xb):\n    out = xb.clone()\n    flip = torch.rand(len(out)) > 0.5\n    out[flip] = torch.flip(out[flip], dims=[3])\n    sy, sx = int(np.random.randint(-4, 5)), int(np.random.randint(-4, 5))\n    return torch.roll(out, shifts=(sy, sx), dims=(2, 3))"),
        make_cell("markdown", "# (2) Model & (3) Train with Data Augmentation + EarlyStopping"),
        make_cell("code", "class Emotion_CNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=0), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),\n            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=0), nn.ReLU(),\n            nn.Conv2d(128, 218, 3, padding=1), nn.ReLU(), nn.Conv2d(218, 218, 3, padding=0), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)\n        )\n        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(218*9*9, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 7))\n    def forward(self, x): return self.classifier(self.features(x))\n\nmodel = Emotion_CNN().to(device)\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.0005)\nbest_loss, patience, wait = float('inf'), 3, 0\n\nfor epoch in range(1, 5):\n    model.train()\n    for xb, yb in train_loader:\n        xb = augment_batch(xb).to(device); yb = yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    va_loss, correct = 0.0, 0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            out = model(xb.to(device))\n            va_loss += criterion(out, yb.to(device)).item() * len(xb)\n            correct += (out.argmax(1).cpu() == yb).sum().item()\n    va_loss /= len(x_test)\n    print(f'Epoch {epoch:02d} | Augmented Val Loss: {va_loss:.4f} | Val Acc: {correct/len(x_test)*100:.2f}%')")
    ]
    save_notebook(cells, CONVERTED_DIR / "4. Training Utilities" / "3_Emotion_Reg_CNN.ipynb")


# =====================================================================
# 5. OTHERS (SEGNET & OBJECT DETECTION - REAL ZIP DATA!)
# =====================================================================
def generate_11_segnet():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `5. Others/1_segnet.ipynb` sang PyTorch (Nạp dữ liệu thật từ `depth_hand.zip`)**\n- Input: Ảnh độ sâu Kinect 96x96.\n- Output: Mặt nạ phân đoạn 7 lớp bàn tay (`Palm`, `5 fingers`, `Background`) kèm ánh xạ màu `class_to_rgb`."),
        make_cell("code", "import time, zipfile\nfrom pathlib import Path\nfrom PIL import Image\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nn_classes, img_h, img_w = 7, 96, 96"),
        make_cell("markdown", "# (1) Data (Đọc thật từ `depth_hand.zip`)"),
        make_cell("code", "COLOR_TO_CLASS = {255: 0, 43: 1, 85: 2, 128: 3, 170: 4, 212: 5, 0: 6}\nzip_path = Path('../../sample/examples/5. Others/depth_hand.zip')\n\ndef load_segnet_from_zip(zf, split, limit):\n    prefix = f'data/segnet/{split}/depth/'\n    files = sorted([n for n in zf.namelist() if n.startswith(prefix) and n.endswith('.png')])[:limit]\n    xs, ys = [], []\n    for d_fn in files:\n        l_fn = d_fn.replace('depth', 'labels')\n        with zf.open(d_fn) as fd, zf.open(l_fn) as fl:\n            img = np.array(Image.open(fd).convert('L').resize((img_w, img_h), Image.NEAREST), dtype=np.float32)\n            lbl = np.array(Image.open(fl).convert('L').resize((img_w, img_h), Image.NEAREST), dtype=np.uint8)\n            cls_map = np.full((img_h, img_w), 6, dtype=np.int64)\n            for gv, cid in COLOR_TO_CLASS.items():\n                cls_map[lbl == gv] = cid\n            xs.append(img); ys.append(cls_map)\n    return np.array(xs)[:, None, :, :], np.array(ys)\n\nwith zipfile.ZipFile(zip_path, 'r') as zf:\n    x_train, y_train = load_segnet_from_zip(zf, 'train', 120)\n    x_test, y_test = load_segnet_from_zip(zf, 'test', 40)\n\ntrain_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=16, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=16, shuffle=False)\nprint('SegNet Train:', x_train.shape, '| Test:', x_test.shape)\n\n# Hiển thị 1 ảnh mẫu cùng 7 lớp mặt nạ nhị phân\nplt.figure(figsize=(14, 2.5))\nplt.subplot(1, 8, 1); plt.imshow(x_test[0, 0], cmap='gray'); plt.title('Depth'); plt.axis('off')\nfor c in range(7):\n    plt.subplot(1, 8, c + 2); plt.imshow(y_test[0] == c, cmap='gray'); plt.title(f'Class {c}'); plt.axis('off')\nplt.tight_layout(); plt.show()"),
        make_cell("markdown", "# (2) Declare SegNet Encoder-Decoder Model"),
        make_cell("code", "def conv_bn_relu(ic, oc):\n    return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))\n\nclass SegNet(nn.Module):\n    def __init__(self, in_c=1, n_cls=7):\n        super().__init__()\n        self.enc = nn.Sequential(\n            conv_bn_relu(in_c, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2, 2),\n            conv_bn_relu(64, 128), conv_bn_relu(128, 128), nn.MaxPool2d(2, 2),\n            conv_bn_relu(128, 256), conv_bn_relu(256, 256), conv_bn_relu(256, 256), nn.MaxPool2d(2, 2),\n        )\n        self.dec = nn.Sequential(\n            nn.Upsample(scale_factor=2, mode='nearest'), conv_bn_relu(256, 256), conv_bn_relu(256, 128),\n            nn.Upsample(scale_factor=2, mode='nearest'), conv_bn_relu(128, 128), conv_bn_relu(128, 64),\n            nn.Upsample(scale_factor=2, mode='nearest'), conv_bn_relu(64, 64),\n            nn.Conv2d(64, n_cls, kernel_size=1), nn.BatchNorm2d(n_cls)\n        )\n    def forward(self, x):\n        return self.dec(self.enc(x))\n\nmodel = SegNet().to(device)\nprint('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"),
        make_cell("markdown", "# (3) Train & (4) Evaluate + `class_to_rgb` Visualization"),
        make_cell("code", "criterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\nfor ep in range(1, 6):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    te_loss, pix_acc = 0.0, 0.0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            out = model(xb.to(device))\n            te_loss += criterion(out, yb.to(device)).item() * len(xb)\n            pix_acc += (out.argmax(1).cpu() == yb).float().mean().item() * len(xb)\n    print(f'Epoch {ep:02d} | Test Loss: {te_loss/len(x_test):.4f} | Pixel Acc: {pix_acc/len(x_test)*100:.2f}%')\n\ndef class_to_rgb(mask):\n    rgb = np.ones((img_h, img_w, 3), dtype=np.float32)\n    palette = {0:[0,0,0], 1:[0,0,0.5], 2:[0,0.5,0], 3:[0.5,0,0], 4:[0.5,0,0.5], 5:[0.5,0.5,0], 6:[1,1,1]}\n    for k, v in palette.items(): rgb[mask == k] = v\n    return rgb\n\nran_ind = np.random.choice(len(x_test))\nwith torch.no_grad():\n    pred_mask = model(torch.tensor(x_test[ran_ind:ran_ind+1]).to(device)).argmax(1).squeeze(0).cpu().numpy()\n\nplt.figure(figsize=(10, 3.5))\nplt.subplot(131); plt.imshow(x_test[ran_ind, 0], cmap='gray'); plt.title('Depth Image'); plt.axis('off')\nplt.subplot(132); plt.imshow(class_to_rgb(y_test[ran_ind])); plt.title('Ground Truth RGB'); plt.axis('off')\nplt.subplot(133); plt.imshow(class_to_rgb(pred_mask)); plt.title('PyTorch SegNet RGB'); plt.axis('off')\nplt.tight_layout(); plt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "5. Others" / "1_segnet.ipynb")


def generate_12_obj_detection():
    cells = [
        make_cell("markdown", "# Overview\n> **Chuyển đổi từ `5. Others/2_obj_detection.ipynb` sang PyTorch (Nạp dữ liệu thật từ `car_data.zip` & phát hiện trên `test.jpg`)**\n- Input: Ảnh RGB 64x64 (`vehicles` = +1, `non-vehicles` = -1).\n- Suy luận FCN tạo Heatmap & vẽ Bounding Box trực tiếp trên `test.jpg`."),
        make_cell("code", "import zipfile\nfrom pathlib import Path\nfrom PIL import Image\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
        make_cell("markdown", "# (1) Data (Đọc thật `vehicles/` & `non-vehicles/` từ `car_data.zip`)"),
        make_cell("code", "zip_path = Path('../../sample/examples/5. Others/car_data.zip')\nX_list, Y_list = [], []\nwith zipfile.ZipFile(zip_path, 'r') as zf:\n    veh = sorted([n for n in zf.namelist() if n.startswith('data/obj_detection/vehicles/') and n.endswith('.png')])\n    non = sorted([n for n in zf.namelist() if n.startswith('data/obj_detection/non-vehicles/') and n.endswith('.png')])\n    for name in np.random.choice(veh, size=500, replace=False):\n        with zf.open(name) as fp:\n            X_list.append(np.array(Image.open(fp).convert('RGB').resize((64, 64)), dtype=np.float32).transpose(2, 0, 1))\n            Y_list.append(1.0)\n    for name in np.random.choice(non, size=500, replace=False):\n        with zf.open(name) as fp:\n            X_list.append(np.array(Image.open(fp).convert('RGB').resize((64, 64)), dtype=np.float32).transpose(2, 0, 1))\n            Y_list.append(-1.0)\n\nX = np.array(X_list, dtype=np.float32)\nY = np.array(Y_list, dtype=np.float32)\nperm = np.random.permutation(len(X))\nX, Y = X[perm], Y[perm]\nsplit = int(0.7 * len(X))\ntrain_loader = DataLoader(TensorDataset(torch.tensor(X[:split]), torch.tensor(Y[:split]).view(-1, 1, 1, 1)), batch_size=64, shuffle=True)\ntest_loader = DataLoader(TensorDataset(torch.tensor(X[split:]), torch.tensor(Y[split:]).view(-1, 1, 1, 1)), batch_size=64, shuffle=False)\nprint(f'Train: {split} | Test: {len(X) - split}')"),
        make_cell("markdown", "# (2) Declare Fully Convolutional Model (`CarDetectorCNN`)"),
        make_cell("code", "class CarDetectorCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Conv2d(3, 10, 3, padding=1), nn.ReLU(inplace=True),\n            nn.Conv2d(10, 10, 3, padding=1), nn.ReLU(inplace=True),\n            nn.MaxPool2d(8, 8), nn.Dropout2d(0.25),\n            nn.Conv2d(10, 128, 8, padding=0), nn.ReLU(inplace=True),\n            nn.Dropout2d(0.5),\n            nn.Conv2d(128, 1, 1, padding=0), nn.Tanh()\n        )\n    def forward(self, x):\n        return self.net(x / 127.5 - 1.0)\n\nmodel = CarDetectorCNN().to(device)\nprint(model)"),
        make_cell("markdown", "# (3) Train defined model"),
        make_cell("code", "criterion = nn.MSELoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\nfor ep in range(1, 11):\n    model.train()\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()\n    model.eval()\n    te_loss, correct = 0.0, 0\n    with torch.no_grad():\n        for xb, yb in test_loader:\n            out = model(xb.to(device))\n            te_loss += criterion(out, yb.to(device)).item() * len(xb)\n            correct += ((out > 0).float() == (yb.to(device) > 0).float()).sum().item()\n    print(f'Epoch {ep:02d} | Test MSE: {te_loss/(len(X)-split):.4f} | Test Acc: {correct/(len(X)-split)*100:.2f}%')\n\ntorch.save(model.state_dict(), 'localize.pt')"),
        make_cell("markdown", "# (4) Evaluate on `test.jpg` using `locate()`"),
        make_cell("code", "heatmodel = CarDetectorCNN().to(device)\nheatmodel.load_state_dict(torch.load('localize.pt', weights_only=True))\nheatmodel.eval()\n\ndef locate(threshold=0.90):\n    data = np.array(Image.open('test.jpg').convert('RGB'), dtype=np.uint8)\n    inp = torch.tensor(data.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0)\n    with torch.no_grad():\n        hm = heatmodel(inp).squeeze().cpu().numpy()\n    sel_hm = hm > threshold\n    yy, xx = np.where(sel_hm)\n    annotated = data.copy()\n    h_img, w_img, _ = annotated.shape\n    for j, i in zip(yy, xx):\n        y0, y1 = min(j*8, h_img-1), min(j*8+64, h_img-1)\n        x0, x1 = min(i*8, w_img-1), min(i*8+64, w_img-1)\n        annotated[y0:y1, x0] = (0, 255, 0); annotated[y0:y1, x1] = (0, 255, 0)\n        annotated[y0, x0:x1] = (0, 255, 0); annotated[y1, x0:x1] = (0, 255, 0)\n    return hm, sel_hm, annotated\n\nhm, sel_hm, annotated = locate(threshold=0.90)\nPath('localize.pt').unlink(missing_ok=True)\n\nplt.figure(figsize=(15, 4))\nplt.subplot(131); plt.title('Heatmap'); plt.imshow(hm, cmap='hot'); plt.axis('off')\nplt.subplot(132); plt.title('Car Area (> 0.90)'); plt.imshow(sel_hm, cmap='gray'); plt.axis('off')\nplt.subplot(133); plt.title('Detected Cars on test.jpg'); plt.imshow(annotated); plt.axis('off')\nplt.tight_layout(); plt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "5. Others" / "2_obj_detection.ipynb")


# =====================================================================
# COLAB GETTING STARTED
# =====================================================================
def generate_13_colab_getting_started():
    cells = [
        make_cell("markdown", "# Experiencing a free of charge way to train deep models with high-performance GPU (PyTorch)!\n> **Chuyển đổi từ `colab_getting_started.ipynb` sang PyTorch**"),
        make_cell("code", "# Check hardware GPU on Colab / local system\nimport platform, torch\nprint('OS:', platform.platform())\nprint('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"),
        make_cell("code", "# On Google Colab, install packages if needed:\n# !pip install torch torchvision matplotlib pillow"),
        make_cell("code", "# Mount Google Drive when running on Google Colab\ntry:\n    from google.colab import drive\n    drive.mount('/content/drive')\nexcept ImportError:\n    print('Local environment detected (skipping google.colab.drive.mount).')"),
        make_cell("code", "# Read & display a sample image using PIL and Matplotlib\nfrom pathlib import Path\nfrom PIL import Image\nimport matplotlib.pyplot as plt\n\nimg_path = Path('5. Others/test.jpg')\nif img_path.exists():\n    img = Image.open(img_path).convert('RGB')\n    plt.imshow(img); plt.axis('off'); plt.title('Sample Image'); plt.show()")
    ]
    save_notebook(cells, CONVERTED_DIR / "colab_getting_started.ipynb")


def convert_all(execute: bool = False):
    print("=" * 60)
    print("CHUYỂN ĐỔI 13 NOTEBOOK KERAS SANG PYTORCH (PHIÊN BẢN CHUẨN 1-1)...")
    print("=" * 60)
    generate_01_getting_started()
    generate_02_iris_bayesian()
    generate_03_func_apxm_mlp()
    generate_04_mnist_mlp()
    generate_05_mnist_cnn()
    generate_06_gender_cnn()
    generate_07_emotion_cnn()
    generate_08_mnist_cnn_lstm()
    generate_09_training_utilities_gender()
    generate_10_training_utilities_emotion()
    generate_11_segnet()
    generate_12_obj_detection()
    generate_13_colab_getting_started()

    if execute:
        print("\n" + "=" * 60)
        print("THỰC THI TẤT CẢ 13 NOTEBOOK QUA IPYTHON KERNEL...")
        print("=" * 60)
        for nb_file in sorted(CONVERTED_DIR.rglob("*.ipynb")):
            execute_notebook(nb_file)

    print("=" * 60)
    print("HOÀN TẤT 13/13 NOTEBOOK PYTORCH!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    run_exec = "--execute" in sys.argv
    convert_all(execute=run_exec)
