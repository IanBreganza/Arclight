import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

dataset_path = r"C:\Users\Ian\Desktop\RPI\Faces4Arclight"

converted = 0
failed = 0

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)
    if not os.path.isdir(person_path):
        continue

    for img_file in os.listdir(person_path):
        if img_file.lower().endswith('.heic'):
            heic_path = os.path.join(person_path, img_file)
            jpg_path = os.path.splitext(heic_path)[0] + '.jpg'

            try:
                img = Image.open(heic_path)
                img = img.convert('RGB')
                img.save(jpg_path, 'JPEG', quality=95)
                os.remove(heic_path)  # delete original
                print(f"  ✓ {person_folder}/{img_file} → .jpg")
                converted += 1
            except Exception as e:
                print(f"  ✗ Failed {img_file}: {e}")
                failed += 1

print(f"\nDone! {converted} converted, {failed} failed")