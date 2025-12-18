#!/bin/bash

# MediTrack - Deployment Script
# Automates setup of Pathway streaming and Aparavi integration

set -e  # Exit on error

echo "🏥 MediTrack Deployment Script"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running in correct directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Are you in the project root?"
    exit 1
fi

print_status "Starting MediTrack deployment..."
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.9+ required. Found: Python $PYTHON_VERSION"
    exit 1
fi
print_status "Python $PYTHON_VERSION detected"
echo ""

# Step 2: Create virtual environment
echo "Step 2: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi
source venv/bin/activate
print_status "Virtual environment activated"
echo ""

# Step 3: Upgrade pip
echo "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet
print_status "Pip upgraded"
echo ""

# Step 4: Install system dependencies
echo "Step 4: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    print_status "Detected Debian/Ubuntu system"
    
    # Check if running as root or with sudo
    if [ "$EUID" -eq 0 ]; then
        apt-get update -qq
        apt-get install -y tesseract-ocr libtesseract-dev libgl1-mesa-glx -qq
    else
        print_warning "System packages require sudo. Running with sudo..."
        sudo apt-get update -qq
        sudo apt-get install -y tesseract-ocr libtesseract-dev libgl1-mesa-glx -qq
    fi
    print_status "Tesseract OCR installed"
elif command -v brew &> /dev/null; then
    print_status "Detected macOS system"
    brew install tesseract
    print_status "Tesseract OCR installed"
else
    print_warning "Could not detect package manager. Please install Tesseract OCR manually."
    print_warning "Visit: https://github.com/tesseract-ocr/tesseract"
fi
echo ""

# Step 5: Install Python dependencies
echo "Step 5: Installing Python packages..."
print_warning "This may take several minutes..."
pip install -r requirements.txt --quiet
print_status "All Python packages installed"
echo ""

# Step 6: Create directory structure
echo "Step 6: Creating directory structure..."
mkdir -p data/{uploads,pathway_outputs,aparavi_outputs,sample_wounds,medical_knowledge}
mkdir -p models/wound_segmentation
mkdir -p logs
print_status "Directories created"
echo ""

# Step 7: Set up environment variables
echo "Step 7: Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# MediTrack Environment Variables
# Generated on $(date)

# OpenAI API Key (for LLM analysis)
OPENAI_API_KEY=sk-your-key-here

# Alternative: Google Gemini
GEMINI_API_KEY=your-gemini-key-here

# Pathway License (if required)
PATHWAY_LICENSE_KEY=your-pathway-license-key

# Aparavi API (if using cloud version)
APARAVI_API_KEY=your-aparavi-key-here

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO

# Database (if needed)
DATABASE_URL=sqlite:///./data/meditrack.db
EOF
    print_status ".env file created"
    print_warning "⚠️  IMPORTANT: Edit .env file with your actual API keys"
else
    print_warning ".env file already exists - skipping"
fi
echo ""

# Step 8: Download sample data (if available)
echo "Step 8: Downloading sample data..."
if [ -f "scripts/download_sample_data.py" ]; then
    python scripts/download_sample_data.py
    print_status "Sample data downloaded"
else
    print_warning "Sample data script not found - skipping"
fi
echo ""

# Step 9: Test imports
echo "Step 9: Testing imports..."
python3 << EOF
try:
    import pathway
    import cv2
    import streamlit
    import pytesseract
    print("✅ All critical imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)
EOF
echo ""

# Step 10: Run tests
echo "Step 10: Running basic tests..."
if [ -d "tests" ]; then
    python -m pytest tests/ -v --tb=short || print_warning "Some tests failed"
    print_status "Tests completed"
else
    print_warning "Tests directory not found - skipping"
fi
echo ""

# Summary
echo ""
echo "================================"
echo "🎉 Deployment Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure API Keys:"
echo "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. Add your wound segmentation model:"
echo "   ${YELLOW}cp your_model.pth models/wound_segmentation/${NC}"
echo ""
echo "3. Start Pathway pipeline (Terminal 1):"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo "   ${YELLOW}python src/pathway_pipeline.py${NC}"
echo ""
echo "4. Launch Streamlit dashboard (Terminal 2):"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo "   ${YELLOW}streamlit run src/app.py${NC}"
echo ""
echo "5. Visit the app at:"
echo "   ${GREEN}http://localhost:8501${NC}"
echo ""
echo "📚 Read INTEGRATION_GUIDE.md for detailed setup instructions"
echo ""
echo "Good luck at Hack With Chicago 2.0! 🚀"
