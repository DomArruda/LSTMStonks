"""Pure-NumPy LSTM regressor with manual backprop-through-time.

Drop-in replacement for the tinygrad-based LSTMCell/LSTMRegressor/train_model/predict
in main.py. No compiled backend, no device selection, no JIT — just NumPy arrays.

Design mirrors the original tinygrad version exactly so behavior (architecture,
init, gate order, loss) stays the same:
  - weight_ih, weight_hh: uniform(-k, k) with k = 1/sqrt(hidden_size)
  - biases: zero-initialized
  - gates packed as [i, f, g, o] (chunk order), same as the tinygrad cell
  - final fc: Linear(hidden_size, 1), applied to the LAST layer's LAST timestep
    hidden state (matches `inp` after the layer loop in the tinygrad version)
  - loss: mean squared error
  - optimizer: Adam, same default betas/eps as tinygrad's nn.optim.Adam
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid (avoids overflow in exp for very negative x).
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class LSTMCellNP:
    """Single LSTM layer, all timesteps handled by LSTMRegressorNP.__call__."""

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = (1.0 / hidden_size) ** 0.5
        # Match tinygrad's Tensor.uniform(4*hidden, in, low=-k, high=k): shape (4H, in).
        self.weight_ih = rng.uniform(-k, k, size=(4 * hidden_size, input_size)).astype(np.float64)
        self.weight_hh = rng.uniform(-k, k, size=(4 * hidden_size, hidden_size)).astype(np.float64)
        self.bias_ih = np.zeros(4 * hidden_size, dtype=np.float64)
        self.bias_hh = np.zeros(4 * hidden_size, dtype=np.float64)

    def params(self):
        return [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]

    def param_names(self):
        return ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]


class LinearNP:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        # Match tinygrad nn.Linear default init: uniform(-k, k), k = 1/sqrt(in_features).
        k = (1.0 / in_features) ** 0.5
        self.weight = rng.uniform(-k, k, size=(out_features, in_features)).astype(np.float64)
        self.bias = rng.uniform(-k, k, size=(out_features,)).astype(np.float64)

    def params(self):
        return [self.weight, self.bias]

    def param_names(self):
        return ["weight", "bias"]


class LSTMRegressorNP:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = [
            LSTMCellNP(input_size if layer == 0 else hidden_size, hidden_size, rng)
            for layer in range(num_layers)
        ]
        self.fc = LinearNP(hidden_size, 1, rng)
        self._cache = None  # populated by forward(), consumed by backward()

    # ---- parameter access (for the optimizer) ----------------------------------
    def parameters(self):
        params = []
        for cell in self.cells:
            params.extend(cell.params())
        params.extend(self.fc.params())
        return params

    # ---- forward ----------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, input_size) -> preds: (batch,)"""
        batch, seq_len, _ = x.shape
        H = self.hidden_size
        L = self.num_layers

        # Cache everything backward() needs: per-layer, per-timestep gate activations,
        # cell states, hidden states, and layer inputs.
        cache = {
            "x": x,
            "h": [np.zeros((L, seq_len + 1, batch, H)) for _ in range(1)][0],
            "c": np.zeros((L, seq_len + 1, batch, H)),
            "i": np.zeros((L, seq_len, batch, H)),
            "f": np.zeros((L, seq_len, batch, H)),
            "g": np.zeros((L, seq_len, batch, H)),
            "o": np.zeros((L, seq_len, batch, H)),
            "layer_in": np.zeros((L, seq_len, batch, max(self.cells[0].input_size, H))),
        }

        layer_input_seq = x  # (batch, seq_len, in_size) for layer 0
        for layer, cell in enumerate(self.cells):
            h_prev = np.zeros((batch, H))
            c_prev = np.zeros((batch, H))
            out_seq = np.zeros((batch, seq_len, H))
            for t in range(seq_len):
                xt = layer_input_seq[:, t, :]
                cache["layer_in"][layer, t, :, : xt.shape[1]] = xt

                gates = xt @ cell.weight_ih.T + cell.bias_ih + h_prev @ cell.weight_hh.T + cell.bias_hh
                i_g, f_g, g_g, o_g = np.split(gates, 4, axis=-1)
                i_a, f_a, g_a, o_a = sigmoid(i_g), sigmoid(f_g), np.tanh(g_g), sigmoid(o_g)

                c_t = f_a * c_prev + i_a * g_a
                h_t = o_a * np.tanh(c_t)

                cache["i"][layer, t] = i_a
                cache["f"][layer, t] = f_a
                cache["g"][layer, t] = g_a
                cache["o"][layer, t] = o_a
                cache["c"][layer, t + 1] = c_t
                cache["h"][layer, t + 1] = h_t

                out_seq[:, t, :] = h_t
                h_prev, c_prev = h_t, c_t

            layer_input_seq = out_seq  # feeds next layer

        last_hidden = cache["h"][L - 1, seq_len]  # (batch, H): last layer, last timestep
        cache["last_hidden"] = last_hidden
        pred = last_hidden @ self.fc.weight.T + self.fc.bias  # (batch, 1)
        cache["pred"] = pred
        cache["seq_len"] = seq_len
        cache["batch"] = batch
        self._cache = cache
        return pred.reshape(-1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ---- backward (manual BPTT) --------------------------------------------------
    def backward(self, y_true: np.ndarray) -> dict:
        """Given cached forward pass + targets, compute dL/dparam for every parameter.
        Loss = mean((pred - y_true) ** 2). Returns {id(param): grad_array}.
        """
        cache = self._cache
        L, H = self.num_layers, self.hidden_size
        seq_len, batch = cache["seq_len"], cache["batch"]

        pred = cache["pred"].reshape(-1)  # (batch,)
        dpred = (2.0 / batch) * (pred - y_true)  # dL/dpred, (batch,)
        dpred = dpred.reshape(batch, 1)

        grads = {id(p): np.zeros_like(p) for p in self.parameters()}

        # --- fc layer ---
        last_hidden = cache["last_hidden"]  # (batch, H)
        grads[id(self.fc.weight)] += dpred.T @ last_hidden  # (1, H)
        grads[id(self.fc.bias)] += dpred.sum(axis=0)  # (1,)
        dh_last = dpred @ self.fc.weight  # (batch, H), grad w.r.t. last layer's last hidden

        # --- BPTT through layers (top layer down to layer 0) ---
        # dh_next_seq[t] holds gradient flowing INTO layer `layer+1`'s input at time t,
        # i.e. d(loss)/d(h_t of this layer), accumulated from the layer above.
        dh_from_above = None  # (batch, seq_len, H) or None for the top layer initially

        for layer in reversed(range(L)):
            cell = self.cells[layer]
            dh_next = np.zeros((batch, H))  # grad w.r.t. h_t from t+1 (within this layer)
            dc_next = np.zeros((batch, H))

            dW_ih = np.zeros_like(cell.weight_ih)
            dW_hh = np.zeros_like(cell.weight_hh)
            db_ih = np.zeros_like(cell.bias_ih)
            db_hh = np.zeros_like(cell.bias_hh)
            dlayer_input = np.zeros((batch, seq_len, cell.input_size))

            for t in reversed(range(seq_len)):
                h_t = cache["h"][layer, t + 1]
                c_t = cache["c"][layer, t + 1]
                c_prev = cache["c"][layer, t]
                i_a = cache["i"][layer, t]
                f_a = cache["f"][layer, t]
                g_a = cache["g"][layer, t]
                o_a = cache["o"][layer, t]
                xt = cache["layer_in"][layer, t, :, : cell.input_size]

                dh = dh_next.copy()
                if layer == L - 1 and t == seq_len - 1:
                    dh += dh_last
                if dh_from_above is not None:
                    dh += dh_from_above[:, t, :]

                tanh_c_t = np.tanh(c_t)
                do = dh * tanh_c_t
                dc = dh * o_a * (1 - tanh_c_t ** 2) + dc_next

                di = dc * g_a
                dg = dc * i_a
                df = dc * c_prev
                dc_prev = dc * f_a

                # Activation derivatives (sigmoid'(z) = a*(1-a); tanh'(z) = 1-a^2).
                di_raw = di * i_a * (1 - i_a)
                df_raw = df * f_a * (1 - f_a)
                dg_raw = dg * (1 - g_a ** 2)
                do_raw = do * o_a * (1 - o_a)

                dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=-1)  # (batch, 4H)

                dW_ih += dgates.T @ xt
                dW_hh += dgates.T @ cache["h"][layer, t]
                db_ih += dgates.sum(axis=0)
                db_hh += dgates.sum(axis=0)

                dxt = dgates @ cell.weight_ih
                dh_prev_from_gates = dgates @ cell.weight_hh

                dlayer_input[:, t, :] = dxt
                dh_next = dh_prev_from_gates
                dc_next = dc_prev

            grads[id(cell.weight_ih)] += dW_ih
            grads[id(cell.weight_hh)] += dW_hh
            grads[id(cell.bias_ih)] += db_ih
            grads[id(cell.bias_hh)] += db_hh

            dh_from_above = dlayer_input  # feeds into layer-1's backward as "grad from above"

        return grads


class AdamNP:
    """Matches tinygrad.nn.optim.Adam defaults: lr, b1=0.9, b2=0.999, eps=1e-8."""

    def __init__(self, params, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = {id(p): np.zeros_like(p) for p in params}
        self.v = {id(p): np.zeros_like(p) for p in params}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        b1, b2, eps, lr = self.b1, self.b2, self.eps, self.lr
        for p in self.params:
            g = grads[id(p)]
            m = self.m[id(p)]
            v = self.v[id(p)]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g ** 2)
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
