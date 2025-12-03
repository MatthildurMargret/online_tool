# Local Testing Guide

This guide will help you test the platform locally before deploying.

## Prerequisites

1. **Python 3.8+** installed
2. **Supabase credentials** (URL and API key)
3. **Alpaca API credentials** (optional, for stock price features)
4. **Google Custom Search API key** (optional, for prospecting features)

## Setup Steps

### 1. Install Python Dependencies

```bash
# Navigate to the project directory
cd /Users/matthildur/Desktop/valuation_comps

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root directory (same level as `requirements.txt`):

```bash
# Required for Supabase (taste tree, deals, etc.)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Optional - for stock price features
ALPACA_API=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret_key

# Optional - for prospecting/search features
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_custom_search_engine_id

# Optional - port (defaults to 5001)
PORT=5001
```

**To get your Supabase credentials:**
1. Go to your Supabase project dashboard
2. Navigate to Settings → API
3. Copy the "Project URL" (SUPABASE_URL)
4. Copy the "anon public" key (SUPABASE_KEY)

### 3. Start the Backend Server

```bash
# Make sure you're in the project root directory
# and your virtual environment is activated

# Navigate to backend directory
cd backend

# Run the Flask app
python app.py
```

You should see output like:
```
============================================================
🚀 Valuation Tool – Cleaned Backend
============================================================
📊 Running on port 5001
============================================================
```

The backend will be running at `http://localhost:5001`

### 4. Access the Frontend

The Flask app serves the frontend automatically. Open your browser and navigate to:

```
http://localhost:5001
```

The frontend HTML file (`frontend/new_index.html`) is served at the root route.

## Testing the Taste Tree Feature

1. **Open the application** in your browser at `http://localhost:5001`
2. **Click "Taste Tree"** in the sidebar
3. **Verify the tree loads** - you should see the hierarchical structure
4. **Test editing:**
   - Click on a category node to expand/collapse it
   - Enter a name in the "montage_lead" input field
   - Click "Save" or press Enter
   - Verify the change persists (check Supabase dashboard or refresh the page)

## Testing Other Features

- **Deals**: Click "Deals" in sidebar - should load deals from Supabase
- **Interesting People**: Click "Interesting People" - should load people data
- **Public Markets**: Click "Public Markets" - requires Alpaca API for stock prices
- **Prospecting**: Click "Prospecting" - requires Google API credentials

## Troubleshooting

### Backend won't start

**Error: "Supabase client not initialized"**
- Check that your `.env` file exists and has `SUPABASE_URL` and `SUPABASE_KEY`
- Make sure the `.env` file is in the project root (not in `backend/`)
- Verify your Supabase credentials are correct

**Error: "Module not found"**
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Verify your virtual environment is activated

**Port already in use**
- Change the PORT in your `.env` file to a different number (e.g., 5002)
- Or kill the process using port 5001:
  ```bash
  lsof -ti:5001 | xargs kill -9
  ```

### Frontend not loading

**404 Error or blank page**
- Make sure the backend server is running
- Check that `frontend/new_index.html` exists
- Verify you're accessing `http://localhost:5001` (or your custom port)

**API calls failing**
- Open browser DevTools (F12) → Network tab
- Check for failed API requests
- Verify the backend is running and accessible
- Check browser console for JavaScript errors

### Taste Tree not loading

**"No taste_tree record found"**
- Make sure you have data in your Supabase `taste_tree` table
- Verify the table has at least one record
- Check that the `data` column contains valid JSONB

**Tree not rendering**
- Open browser DevTools console (F12)
- Look for JavaScript errors
- Check the Network tab to see if `/api/taste-tree` request succeeded

## Quick Test Checklist

- [ ] Backend server starts without errors
- [ ] Frontend loads at `http://localhost:5001`
- [ ] Sidebar shows all modules including "Taste Tree"
- [ ] Taste Tree module loads and displays the tree structure
- [ ] Can expand/collapse tree nodes
- [ ] Can edit montage_lead field
- [ ] Changes save successfully (check Supabase or refresh page)
- [ ] Search/filter works in Taste Tree

## Development Tips

1. **Hot Reload**: The Flask app runs in debug mode, so backend changes will auto-reload. Frontend changes require a browser refresh.

2. **Database Changes**: If you modify the Supabase schema, you may need to update the backend code accordingly.

3. **Testing Taste Tree Updates**: After updating a montage_lead, check your Supabase dashboard to verify the change was saved to the database.

4. **Console Logging**: Check both:
   - Backend terminal for server-side errors
   - Browser DevTools console for client-side errors

## Next Steps

Once local testing is successful:
1. Test all major features
2. Verify data persistence in Supabase
3. Check for any console errors
4. Test on different browsers if needed
5. Ready to deploy!

