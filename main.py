from torch.utils.data import DataLoader
import wandb
from dataloaders.builder import get_dataloaders
from models.builder import get_model
from models.trainer import train, test
import torch
import torch.nn as nn
import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model_name", type=str, default="UNet") #DeepLabV3
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--backbone_name", type=str, default="densenet121") #dense121 resnet50
    parser.add_argument("--output_dir", type=str, default="output/")
    return parser.parse_args()
def main():
    args = parse_config()
    wandb.init(project="semantic-segmentation-SUIM_v1", config=vars(args))
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.p)
    model = get_model(args.model_name, args.num_classes, backbone_name=args.backbone_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, args.model_name, args.backbone_name, train_loader, val_loader, criterion, optimizer, args.num_epochs, args.batch_size, args.learning_rate, args.p, args.output_dir)
    test(model, args.model_name, test_loader, suffix="final")
    model_best = model
    if args.model_name == "UNet":
        model_best.load_state_dict(torch.load(args.output_dir + f"model_best_unet_{args.backbone_name}_{args.batch_size}_{args.num_epochs}_{args.learning_rate}_{args.p}.pth"))
    elif args.model_name == "DeepLabV3":
        model_best.load_state_dict(torch.load(args.output_dir + f"model_best_deeplab_{args.batch_size}_{args.num_epochs}_{args.learning_rate}_{args.p}.pth"))
    test(model_best, args.model_name, test_loader,  suffix="best")
if __name__ == "__main__":
    main()
