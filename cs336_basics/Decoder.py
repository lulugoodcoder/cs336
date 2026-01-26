import torch
class Decoder:
    def decode(
        self,
        model,
        tokenizer,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,  # 1.0 means no filtering
        device: str = "cpu",
    ) -> str:
        # Put model in evaluation mode (disables dropout, etc.)
        model.eval()
    
        # Get the end-of-text token ID from the tokenizer
        eos_token_id = tokenizer.eos_token_id 
        
        # 1. Tokenize the prompt
        input_ids = tokenizer.encode(prompt)  # list of integers
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # 1. Run model to get logits
                input_tensor = torch.tensor([input_ids], device=device)  # add batch dimension
                logits = model(input_tensor)            # get all predictions
                next_logits = logits[0, -1, :]          # get last position only (batch, seq, vocab) 

                # 2. Temperature scaling + convert to probabilities
                probs = torch.softmax(next_logits / temperature, dim = -1)

                if top_p < 1.0:
                    probs = self.apply_top_p(probs, top_p)
                
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == eos_token_id:
                    break

                input_ids.append(next_token)
        
        return tokenizer.decode(input_ids)

    def apply_top_p(self, probs, top_p):
        sorted_probs, sorted_indices = torch.sort(probs, dim = -1, descending=True)

        cumsum = torch.cumsum(sorted_probs, dim=-1)

        mask = cumsum > top_p

        mask[0] = False

        sorted_probs[mask] = 0.0

        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        return probs / probs.sum()