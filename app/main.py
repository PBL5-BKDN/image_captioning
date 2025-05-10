from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Simple FastAPI Server")

# Define a Pydantic model for request body
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    quantity: int

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server!"}

# Get item by ID
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "message": f"Item {item_id} retrieved"}

# Create new item
@app.post("/items/")
async def create_item(item: Item):
    return {
        "message": "Item created successfully",
        "item": item.dict()
    }

# Update item
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {
        "message": f"Item {item_id} updated successfully",
        "item": item.dict()
    }

# Delete item
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"message": f"Item {item_id} deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)