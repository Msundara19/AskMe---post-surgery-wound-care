from meditrack.config import CONFIG

def main():
    CONFIG.sample_wounds_dir.mkdir(parents=True, exist_ok=True)
    print("Sample data directories ensured.")

if __name__ == "__main__":
    main()
