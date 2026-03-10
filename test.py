from deepface import DeepFace
import os

# Paths
known_faces_dir = "known_faces"
test_images_dir = "test_images"

# Loop through each test image
for test_image in os.listdir(test_images_dir):
    test_image_path = os.path.join(test_images_dir, test_image)

    print(f"\n🔍 Checking {test_image}...")
    matched = False

    # Compare with each known face
    for known_face in os.listdir(known_faces_dir):
        known_face_path = os.path.join(known_faces_dir, known_face)

        try:
            result = DeepFace.verify(
                img1_path=test_image_path,
                img2_path=known_face_path,
                model_name="Facenet",
                enforce_detection=False
            )

            if result["verified"]:
                print(f"✅ Match found: {test_image} matches with {known_face}")
                matched = True
                break
        except Exception as e:
            print(f"⚠ Error processing {known_face}: {e}")

    if not matched:
        print(f"❌ No match found for {test_image}")
