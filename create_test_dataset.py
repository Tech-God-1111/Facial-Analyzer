"""
Quick script to create a test dataset structure
"""

import os
from pathlib import Path


def create_test_dataset():
    """Create a minimal test dataset structure"""
    base_dir = Path("datasets")

    # Create directories
    folders = ["train", "test", "validation", "gallery"]
    persons = ["person1", "person2", "person3"]

    for folder in folders:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        for person in persons:
            person_path = folder_path / person
            person_path.mkdir(exist_ok=True)

            # Create placeholder files
            for i in range(1, 4):
                placeholder = person_path / f"{person}_{i:03d}.txt"
                placeholder.write_text(f"Placeholder for {person} image {i}\n"
                                       f"Replace with actual .jpg/.png images")

    print(f"✅ Dataset structure created at: {base_dir}")
    print(f"📁 Created {len(folders)} folders with {len(persons)} persons each")
    print(f"📝 Add actual image files (jpg/png) in place of the .txt files")


if __name__ == "__main__":
    create_test_dataset()