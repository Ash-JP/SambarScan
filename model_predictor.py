from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random

model = load_model("sambar_model.keras")
classes = ["watery", "goodsambar"]

reactions = {
    "watery": [
        "💧 Did you mean to make rasam?",
        "🫗 Dosa just left the group.",
        "🚿 Your sambar is legally a beverage.",
        "👀 Add some sambar to your water, not the other way around!",
        "📉 Viscosity: Low. Hopes: Lower.",
        "🥲 Amma’s ghost just blinked.",
        "😅 Do you store this in a water bottle?",
        "🫣 The sambar tried to escape the plate!"
    ],
    "goodsambar": [
        "🌟 That’s a masterclass in consistency.",
        "🍛 Michelin-star sambar detected!",
        "💪 This sambar holds flavor and dignity.",
        "🧘 Balanced. Soulful. Divinely spiced.",
        "🎖️ Amma just said ‘good job, da’ from the kitchen realm.",
        "📚 Add this to history books under ‘Perfection’",
        "🏆 Your sambar just graduated top of its class.",
        "😌 This sambar slaps. And then hugs you after."
    ]
}

memes = [
    "📡 NASA has been notified.",
    "🧂 Model accuracy depends on spice level.",
    "📷 Warning: sambar might become an influencer.",
    "🍽️ Serve with respect. This ain’t no ordinary lunch.",
    "👨‍🍳 Sambar just earned a LinkedIn badge: Consistency Pro.",
    "🤖 AI can detect thickness, but not your tears.",
    "📦 Box it, brand it, launch the startup: SambarScan™",
    "🎬 Coming soon: *The Sambar Redemption*"
]

def check_sambar_consistency(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    label = classes[predicted_index]

    main_joke = random.choice(reactions[label])
    meme_side = random.choice(memes)

    result_text = f"""
        🍛 <b>{label.upper()}</b><br>
        {main_joke}<br>
        <i>{meme_side}</i>
    """

    return result_text, list(prediction), classes
