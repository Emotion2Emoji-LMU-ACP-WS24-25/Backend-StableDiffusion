import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from pymongo import MongoClient

SDXL_PATH = "/app/models/stable-diffusion-xl-base-1.0"
EMOJI_XL_PATH = "/app/models/ios_emoji_xl_v2_lora.safetensors"
MONGO_URI = "mongodb://mongodb:27017"
DB_NAME = "imageUploadDB"

# Load StableDiffusion pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_PATH, torch_dtype=torch.float16, local_files_only=True).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(EMOJI_XL_PATH, lora_scale=0.6)

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

try:
    # Test the connection by checking the server status
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print("MongoDB connection failed:", e)

def get_pending_jobs():
    """
    This function retrieves all entries with the status 'prompt_created' from all collections,
    sorted by creation date (oldest first).
    """
    pending_jobs = []

    # Iterate through all collections in the database
    for collection_name in db.list_collection_names():
        collection = db[collection_name]

        # Find all entries with the status 'prompt_created' and sort by creation date
        jobs = collection.find({"status": "prompted"}).sort("uploadDate", 1)  # 1 for ascending order
        # Add the found jobs to the pending jobs list
        for job in jobs:
            pending_jobs.append({
                "collection": collection_name,
                "entry": job
            })

    # Sort the entire list of pending jobs by creation date
    pending_jobs.sort(key=lambda x: x["entry"]["uploadDate"])

    return pending_jobs


def generate_emojis():
    jobs = get_pending_jobs()
    if not jobs:
        print("No pending jobs!")
        return

    while jobs:  # Continue processing until the job list is empty
        oldest_job = jobs.pop(0)  # Remove and get the first job from the list
        print(f"Processing job: {oldest_job}")
        
        collection_name = oldest_job['collection']
        collection = db[collection_name]

        # Update the status in the database to "create_image"
        collection.update_one(
            {"_id": oldest_job['entry']['_id']}, 
            {"$set": {"status": "create_image"}}
        )

        try:
            # Extract prompt
            prompt = oldest_job['entry']["prompt"]
            print("Generating image with prompt:", prompt)

            # Update the status in the database to "generating"
            collection.update_one(
                {"_id": oldest_job['entry']['_id']}, 
                {"$set": {"status": "generating"}}
            )
            
            # Attempt to generate the image
            image = pipe(prompt, negative_prompt="blurry").images[0]
            image_path = f"/app/emojis/{oldest_job['entry']['_id']}.png"
            image.save(image_path)
            print(f"Image saved at {image_path}")

            # Update the status to "completed" and update the resultImagePath
            collection.update_one(
                {"_id": oldest_job['entry']['_id']},
                {"$set": {"status": "completed", "resultImagePath": image_path}}
            )
            print("Status updated to 'image_generated'")

        except Exception as e:
            print(f"Error generating image for job {oldest_job['entry']['_id']}: {e}")
        
# Loop, 15s intervals
while True:
    try:
        generate_emojis()
    except Exception as e:
        print("Error: ", e)
    print("Warte 15 Sekunden...")
    time.sleep(15)