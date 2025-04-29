import torch


def evaluate_model(model, test_chart_loader, test_num_loader=None):
    

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return [move_to_device(i) for i in x]
        else:
            return x

    # Check if multimodal
    if isinstance(test_chart_loader, list):  
        chart_loader_branches = test_chart_loader
        is_multimodal = True
    else:
        chart_loader_branches = [test_chart_loader]
        is_multimodal = False

    # Setup iterator
    if test_num_loader:
        iterator = zip(zip(*chart_loader_branches), test_num_loader)
    else:
        iterator = zip(*chart_loader_branches)

    with torch.inference_mode():
        for batch in iterator:
            if test_num_loader:
                chart_batch, num_batch = batch
                chart_inputs = move_to_device(chart_batch)
                num_features, labels = move_to_device(num_batch)
                chart_imgs = [img for img, _ in chart_inputs]

                outputs = model((chart_imgs, num_features))   # 🚨 KEY

            else:
                if is_multimodal:
                    chart_inputs = move_to_device(batch)
                    labels = chart_inputs[0][1]
                    chart_imgs = [img for img, _ in chart_inputs]

                    outputs = model(chart_imgs)   # 🚨 KEY
                else:
                    images, labels = move_to_device(batch)

                    outputs = model(images)   # 🚨 KEY

            labels = labels.to(device)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(chart_loader_branches[0])
    test_accuracy = 100 * correct / total

    print(f"\n=== Evaluation Results ===")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

