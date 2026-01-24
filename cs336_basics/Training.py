import torch
import numpy as np
import argparse
from cs336_basics.DataLoader import DataLoader
from cs336_basics.TansformerLM import TansformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.CrossEntropy import CrossEntropy
from cs336_basics.CheckPoint import CheckPoint
from cs336_basics.LearningRateSchedule import LearningRateSchedule
from cs336_basics.GradientClipping import GradientClipping

class Training:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Data args
        parser.add_argument('--train_data', type=str, required=True)
        parser.add_argument('--val_data', type=str, required=True)

        # Model args
        parser.add_argument('--vocab_size', type=int, default=10000)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--num_heads', type=int, default=16)
        parser.add_argument('--d_ff', type=int, default=1344)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--context_length', type=int, default=256)
        parser.add_argument('--rope_theta', type=float, default=10000.0)

                # Training args
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--max_steps', type=int, default=5000)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--min_lr', type=float, default=1e-4)
        parser.add_argument('--warmup_steps', type=int, default=100)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        
        # Logging args
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--val_interval', type=int, default=500)
        parser.add_argument('--checkpoint_interval', type=int, default=1000)
        parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt')
        
        # Device
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.args = parser.parse_args()

    def load_data(self, path) -> np.ndarray:
        return np.memmap(path, dtype=np.uint16, mode='r')
    
    def compute_loss(self, model, inputs, targets, loss_fn):
        """Forward pass and compute loss"""
        logits = model(inputs)
        return loss_fn(logits, targets)

    def get_batch(self, data: np.ndarray):
        """Get a random batch from data"""
        data_loader = DataLoader()
        return data_loader.load(data, self.args.batch_size, self.args.context_length, self.args.device)

    @torch.no_grad()
    def evaluate(self, model, val_data, loss_fn, num_batches=10):
        model.eval()
        total_loss = 0.0

        for _ in range(num_batches):
            inputs, targets = self.get_batch(val_data)  # Random is OK
            loss = self.compute_loss(model, inputs, targets, loss_fn)
            total_loss += loss.item()

        model.train()
        return total_loss / num_batches

    def train(self):
        model = TansformerLM(
            self.args.vocab_size, self.args.d_model, self.args.num_heads, 
            self.args.d_ff, self.args.num_layers, self.args.context_length, 
            self.args.rope_theta, self.args.device
        ).to(self.args.device)

        # Fixed: model.parameters()
        optimizer = AdamW(model.parameters(), lr=self.args.lr)
        loss_fn = CrossEntropy()
        gradient_clipping = GradientClipping()
        lr_schedule = LearningRateSchedule()

        training_data = self.load_data(self.args.train_data)
        validation_data = self.load_data(self.args.val_data)

        model.train()

        for step in range(self.args.max_steps):
            inputs, targets = self.get_batch(training_data) 

            lr = lr_schedule.learning_rate_schedule(
                step, 
                self.args.lr, 
                self.args.min_lr, 
                self.args.warmup_steps, 
                self.args.max_steps
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            loss = self.compute_loss(model, inputs, targets, loss_fn)

            loss.backward()

            gradient_clipping.gradient_clipping(model.parameters(), self.args.max_grad_norm)

            optimizer.step()

            # Add logging (optional but helpful)
            if step % self.args.log_interval == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

        



    
        

    