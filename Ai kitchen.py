import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Updated dataset of recipes with additional categories
recipes = pd.DataFrame({
    'recipe': ['Burger', 'Stir Fry', 'Lasagna', 'Salmon', 'Smoothie'],
    'ingredients': ['bun, patty, lettuce, tomato', 'tofu, broccoli, soy sauce', 'pasta, cheese, tomato sauce', 'salmon fillet, lemon, dill', 'banana, spinach, yogurt'],
    'dietary': ['non-vegetarian', 'vegan', 'vegetarian', 'pescatarian', 'vegetarian']
})

# Updated user preferences with dietary restrictions and allergies
user_preferences = {
    'dietary': 'vegan',
    'favorite_ingredients': ['tofu', 'broccoli'],
    'allergies': ['nuts', 'dairy']
}

# Vectorize ingredients
ingredients_vector = recipes['ingredients'].str.get_dummies(sep=', ')

# Train a KNN model for recipe recommendation
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(ingredients_vector)

def recommend_recipe(user_ingredients):
    # Vectorize user ingredients for personalized recipe recommendation
    user_vector = pd.Series([user_ingredients]).str.get_dummies(sep=', ')
    
    # Align with recipe ingredients vector
    user_vector = user_vector.reindex(columns=ingredients_vector.columns, fill_value=0)
    
    # Recommend a recipe based on dietary preferences
    distances, indices = knn.kneighbors(user_vector)
    recommended_recipe = recipes.iloc[indices[0]]
    return recommended_recipe['recipe'].values[0]

# Load pre-trained model for utensil detection (using ResNet50)
model = ResNet50(weights='imagenet')

def detect_utensil(image_path):
    # Load and preprocess utensil image for detection
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Predict utensil class using ResNet50 model
    preds = model.predict(x)
    predictions = tf.keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    utensils = [(pred[1], pred[2]) for pred in predictions]
    return utensils, img  # Return detected utensils and the original image

class Inventory:
    def __init__(self):
        # Initialize inventory with available ingredients and quantities
        self.ingredients = {
            'bun': 15,
            'patty': 10,
            'lettuce': 20,
            'tomato': 18,
            'tofu': 12,
            'broccoli': 8,
            'pasta': 10,
            'cheese': 5,
            'tomato sauce': 7,
            'salmon fillet': 3,
            'lemon': 4,
            'dill': 2,
            'banana': 6,
            'spinach': 5,
            'yogurt': 8
        }

    def check_ingredients(self, required_ingredients):
        # Check availability of required ingredients in inventory
        missing_ingredients = []
        for ingredient, quantity in required_ingredients.items():
            if self.ingredients.get(ingredient, 0) < quantity:
                missing_ingredients.append(ingredient)
        return missing_ingredients

class Oven:
    def __init__(self):
        # Initialize oven with default settings
        self.temperature = 0
        self.timer = 0

    def set_temperature(self, temp):
        # Set oven temperature for cooking
        self.temperature = temp
        print(f"Oven temperature set to {temp}Â°C")

    def set_timer(self, time):
        # Set timer for cooking duration
        self.timer = time
        print(f"Oven timer set to {time} minutes")

# Function to generate and save image of detected utensils
def save_detected_utensil_image(utensil_img, utensil_name):
    save_dir = 'D:/detected_utensils'  # Adjust path as needed
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it does not exist
    save_path = os.path.join(save_dir, f"{utensil_name}.jpg")
    utensil_img.save(save_path)
    print(f"Detected utensil image saved: {save_path}")

# Simulate AI Kitchen with enhanced features
def ai_kitchen(user_ingredients, utensil_image_path):
    try:
        # Step 1: Recommend a recipe based on user preferences
        recommended_recipe = recommend_recipe(user_ingredients)
        print(f"Recommended Recipe: {recommended_recipe}")
    except Exception as e:
        print(f"Error in recommending recipe: {e}")
        return
    
    try:
        # Step 2: Detect utensil from image using ResNet50 model
        utensils, utensil_img = detect_utensil(utensil_image_path)
        print("Detected Utensils:")
        for idx, utensil in enumerate(utensils, start=1):
            print(f"{idx}. {utensil[0]} (confidence: {utensil[1]:.2f})")
    except Exception as e:
        print(f"Error in detecting utensil: {e}")
        return
    
    try:
        # Step 3: Save image of detected utensil
        for utensil in utensils:
            save_detected_utensil_image(utensil_img, utensil[0])
    except Exception as e:
        print(f"Error in saving detected utensil image: {e}")
        return
    
    try:
        # Step 4: Check inventory for required ingredients
        inventory = Inventory()
        required_ingredients = {ing: 1 for ing in user_ingredients.split(', ')}  # Assume equal quantities for simplicity
        missing = inventory.check_ingredients(required_ingredients)
        if missing:
            print(f"Missing ingredients: {', '.join(missing)}")
        else:
            print("All ingredients are available in inventory.")
    except Exception as e:
        print(f"Error in checking inventory: {e}")
        return
    
    try:
        # Step 5: Use oven to cook the recommended recipe
        oven = Oven()
        oven.set_temperature(180)  # Set oven temperature based on recipe requirements
        oven.set_timer(30)  # Set timer based on recipe cooking time
    except Exception as e:
        print(f"Error in using the oven: {e}")

# Example usage with updated parameters
user_ingredients = 'tofu, broccoli, soy sauce'
utensil_image_path = 'D:/utensil.jpeg'  # Replace with the actual image path
ai_kitchen(user_ingredients, utensil_image_path)
