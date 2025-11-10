import torch
import argparse
from mediapipe_train import SignLanguageModel  

def export_torchscript_model(checkpoint_path, output_path, input_size=477, sequence_length=64, num_classes=25):
    model = SignLanguageModel(
        input_size=input_size,
        hidden_size=2048,
        num_classes=num_classes,
        dropout=0.2
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ✅ Use scripting instead of tracing
    scripted_model = torch.jit.script(model)

    scripted_model.save(output_path)
    print(f"✅ TorchScript model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", default="sign_language_model.pt")
    parser.add_argument("--input_size", type=int, default=477)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=25)
    args = parser.parse_args()

    export_torchscript_model(
        args.checkpoint_path,
        args.output_path,
        args.input_size,
        args.sequence_length,
        args.num_classes
    )
