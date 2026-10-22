from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from torch.utils.data import DataLoader


# DatasetDict({
#    train: Dataset({
#        features: ['observation.image', 'observation.state', 'action', 'episode_index',
#                    'frame_index', 'timestamp', 'next.reward', 'n
#ext.done', 'next.success', 'index', 'task_index'],
#        num_rows: 25650
#    })
#})

ds = load_dataset("lerobot/pusht_image")
dataset = ds["train"]
dataset.set_format("torch", columns=["observation.image", "observation.state", "action"])

episode_id = 10

# Get frames only from this episode 
episodes = ds["train"].filter(
        lambda x : x["episode_index"] == episode_id
)

fig, ax = plt.subplots(figsize=(12,12))

def update(i):
    ax.clear()
    ax.imshow(episodes[i]["observation.image"])
    ax.axis("off")

animation = FuncAnimation(
        fig,
        update,
        frames = len(episodes),
        interval = 50, # miliseconds between frames
        repeat=False
)

# plt.show()

batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

for batch in data_loader:
    observations = batch["observation.state"].to(device) # (16,2)
    actions = batch["action"].to(device)   # (16,2)
    images = batch["observation.image"].to(device) # (16, 3, 96, 96)
    
    print(observations.shape)
    print(actions.shape)
    print(images.shape)
