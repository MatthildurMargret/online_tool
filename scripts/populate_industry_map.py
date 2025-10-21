#!/usr/bin/env python3
"""
Script to populate industry_map.json with stock tickers from S&P 500 CSV
"""

import json
import csv
from pathlib import Path

# Define the mapping from CSV "GICS Sub-Industry" to our JSON structure
# Format: "CSV Sub-Industry Name": ("Industry", "Sector", "Industry Group")
INDUSTRY_MAPPING = {
    # Communication Services
    "Alternative Carriers": ("Communication Services", "Diversified Telecommunication Services", "Alternative Carriers"),
    "Integrated Telecommunication Services": ("Communication Services", "Diversified Telecommunication Services", "Integrated Telecommunication Services"),
    "Wireless Telecommunication Services": ("Communication Services", "Diversified Telecommunication Services", "Wireless Telecommunication Services"),
    "Advertising": ("Communication Services", "Media & Entertainment", "Advertising"),
    "Broadcasting": ("Communication Services", "Media & Entertainment", "Broadcasting"),
    "Cable & Satellite": ("Communication Services", "Media & Entertainment", "Cable & Satellite"),
    "Movies & Entertainment": ("Communication Services", "Media & Entertainment", "Movies & Entertainment"),
    "Interactive Media & Services": ("Communication Services", "Media & Entertainment", "Interactive Media & Services"),
    "Interactive Home Entertainment": ("Communication Services", "Media & Entertainment", "Movies & Entertainment"),
    "Publishing": ("Communication Services", "Media & Entertainment", "Broadcasting"),
    
    # Consumer Discretionary
    "Automotive Parts & Equipment": ("Consumer Discretionary", "Automobiles & Components", "Automotive Parts & Equipment"),
    "Tires & Rubber": ("Consumer Discretionary", "Automobiles & Components", "Tires & Rubber"),
    "Automobile Manufacturers": ("Consumer Discretionary", "Automobiles & Components", "Automobile Manufacturers"),
    "Motorcycle Manufacturers": ("Consumer Discretionary", "Automobiles & Components", "Motorcycle Manufacturers"),
    "Household Appliances": ("Consumer Discretionary", "Consumer Durables & Apparel", "Household Appliances"),
    "Housewares & Specialties": ("Consumer Discretionary", "Consumer Durables & Apparel", "Housewares & Specialties"),
    "Leisure Products": ("Consumer Discretionary", "Consumer Durables & Apparel", "Leisure Products"),
    "Textiles, Apparel & Luxury Goods": ("Consumer Discretionary", "Consumer Durables & Apparel", "Textiles, Apparel & Luxury Goods"),
    "Apparel, Accessories & Luxury Goods": ("Consumer Discretionary", "Consumer Durables & Apparel", "Textiles, Apparel & Luxury Goods"),
    "Footwear": ("Consumer Discretionary", "Consumer Durables & Apparel", "Textiles, Apparel & Luxury Goods"),
    "Consumer Electronics": ("Consumer Discretionary", "Consumer Durables & Apparel", "Housewares & Specialties"),
    "Hotels, Resorts & Cruise Lines": ("Consumer Discretionary", "Consumer Services", "Hotels, Resorts & Cruise Lines"),
    "Restaurants": ("Consumer Discretionary", "Consumer Services", "Restaurants"),
    "Leisure Facilities": ("Consumer Discretionary", "Consumer Services", "Leisure Facilities"),
    "Casinos & Gaming": ("Consumer Discretionary", "Consumer Services", "Leisure Facilities"),
    "Education Services": ("Consumer Discretionary", "Consumer Services", "Education Services"),
    "Specialized Consumer Services": ("Consumer Discretionary", "Consumer Services", "Leisure Facilities"),
    "Distributors": ("Consumer Discretionary", "Retailing", "Distributors"),
    "Broadline Retail": ("Consumer Discretionary", "Retailing", "Broadline Retail"),
    "Apparel Retail": ("Consumer Discretionary", "Retailing", "Apparel Retail"),
    "Computer & Electronics Retail": ("Consumer Discretionary", "Retailing", "Computer & Electronics Retail"),
    "Home Improvement Retail": ("Consumer Discretionary", "Retailing", "Home Improvement Retail"),
    "Other Specialty Retail": ("Consumer Discretionary", "Retailing", "Other Specialty Retail"),
    "Automotive Retail": ("Consumer Discretionary", "Retailing", "Automotive Retail"),
    "Homefurnishing Retail": ("Consumer Discretionary", "Retailing", "Homefurnishing Retail"),
    "Home Furnishings": ("Consumer Discretionary", "Retailing", "Homefurnishing Retail"),
    "Homebuilding": ("Consumer Discretionary", "Consumer Services", "Leisure Facilities"),
    
    # Consumer Staples
    "Drug Retail": ("Consumer Staples", "Food & Staples Retailing", "Drug Retail"),
    "Food Distributors": ("Consumer Staples", "Food & Staples Retailing", "Food Distributors"),
    "Food Retail": ("Consumer Staples", "Food & Staples Retailing", "Food Retail"),
    "Consumer Staples Merchandise Retail": ("Consumer Staples", "Food & Staples Retailing", "Consumer Staples Merchandise Retail"),
    "Agricultural Products & Services": ("Consumer Staples", "Food, Beverage & Tobacco", "Agricultural Products"),
    "Agricultural Products": ("Consumer Staples", "Food, Beverage & Tobacco", "Agricultural Products"),
    "Brewers": ("Consumer Staples", "Food, Beverage & Tobacco", "Brewers"),
    "Distillers & Vintners": ("Consumer Staples", "Food, Beverage & Tobacco", "Distillers & Vintners"),
    "Food Products": ("Consumer Staples", "Food, Beverage & Tobacco", "Food Products"),
    "Packaged Foods & Meats": ("Consumer Staples", "Food, Beverage & Tobacco", "Food Products"),
    "Soft Drinks & Non-alcoholic Beverages": ("Consumer Staples", "Food, Beverage & Tobacco", "Soft Drinks"),
    "Soft Drinks": ("Consumer Staples", "Food, Beverage & Tobacco", "Soft Drinks"),
    "Tobacco": ("Consumer Staples", "Food, Beverage & Tobacco", "Tobacco"),
    "Household Products": ("Consumer Staples", "Household & Personal Products", "Household Products"),
    "Personal Care Products": ("Consumer Staples", "Household & Personal Products", "Personal Care Products"),
    
    # Energy
    "Oil & Gas Drilling": ("Energy", "Energy Equipment & Services", "Oil & Gas Drilling"),
    "Oil & Gas Equipment & Services": ("Energy", "Energy Equipment & Services", "Oil & Gas Equipment & Services"),
    "Integrated Oil & Gas": ("Energy", "Oil, Gas & Consumable Fuels", "Integrated Oil & Gas"),
    "Oil & Gas Exploration & Production": ("Energy", "Oil, Gas & Consumable Fuels", "Oil & Gas Exploration & Production"),
    "Oil & Gas Refining & Marketing": ("Energy", "Oil, Gas & Consumable Fuels", "Oil & Gas Refining & Marketing"),
    "Oil & Gas Storage & Transportation": ("Energy", "Oil, Gas & Consumable Fuels", "Oil & Gas Storage & Transportation"),
    "Coal & Consumable Fuels": ("Energy", "Oil, Gas & Consumable Fuels", "Coal & Consumable Fuels"),
    
    # Financials
    "Diversified Banks": ("Financials", "Banks", "Diversified Banks"),
    "Regional Banks": ("Financials", "Banks", "Regional Banks"),
    "Diversified Financial Services": ("Financials", "Diversified Financials", "Diversified Financial Services"),
    "Multi-Sector Holdings": ("Financials", "Diversified Financials", "Multi-Sector Holdings"),
    "Specialized Finance": ("Financials", "Diversified Financials", "Specialized Finance"),
    "Consumer Finance": ("Financials", "Diversified Financials", "Consumer Finance"),
    "Asset Management & Custody Banks": ("Financials", "Diversified Financials", "Asset Management & Custody Banks"),
    "Investment Banking & Brokerage": ("Financials", "Diversified Financials", "Investment Banking & Brokerage"),
    "Transaction & Payment Processing Services": ("Financials", "Diversified Financials", "Specialized Finance"),
    "Financial Exchanges & Data": ("Financials", "Diversified Financials", "Specialized Finance"),
    "Insurance Brokers": ("Financials", "Insurance", "Insurance Brokers"),
    "Life & Health Insurance": ("Financials", "Insurance", "Life & Health Insurance"),
    "Property & Casualty Insurance": ("Financials", "Insurance", "Property & Casualty Insurance"),
    "Multi-line Insurance": ("Financials", "Insurance", "Property & Casualty Insurance"),
    "Reinsurance": ("Financials", "Insurance", "Reinsurance"),
    
    # Health Care
    "Health Care Equipment": ("Health Care", "Health Care Equipment & Services", "Health Care Equipment"),
    "Health Care Supplies": ("Health Care", "Health Care Equipment & Services", "Health Care Supplies"),
    "Health Care Distributors": ("Health Care", "Health Care Equipment & Services", "Health Care Distributors"),
    "Health Care Services": ("Health Care", "Health Care Equipment & Services", "Health Care Services"),
    "Health Care Facilities": ("Health Care", "Health Care Equipment & Services", "Health Care Facilities"),
    "Managed Health Care": ("Health Care", "Health Care Equipment & Services", "Managed Health Care"),
    "Health Care Technology": ("Health Care", "Health Care Equipment & Services", "Health Care Technology"),
    "Biotechnology": ("Health Care", "Pharmaceuticals, Biotechnology & Life Sciences", "Biotechnology"),
    "Pharmaceuticals": ("Health Care", "Pharmaceuticals, Biotechnology & Life Sciences", "Pharmaceuticals"),
    "Life Sciences Tools & Services": ("Health Care", "Pharmaceuticals, Biotechnology & Life Sciences", "Life Sciences Tools & Services"),
    
    # Industrials
    "Aerospace & Defense": ("Industrials", "Capital Goods", "Aerospace & Defense"),
    "Building Products": ("Industrials", "Capital Goods", "Building Products"),
    "Construction & Engineering": ("Industrials", "Capital Goods", "Construction & Engineering"),
    "Electrical Components & Equipment": ("Industrials", "Capital Goods", "Electrical Equipment"),
    "Electrical Equipment": ("Industrials", "Capital Goods", "Electrical Equipment"),
    "Industrial Conglomerates": ("Industrials", "Capital Goods", "Industrial Conglomerates"),
    "Machinery": ("Industrials", "Capital Goods", "Machinery"),
    "Construction Machinery & Heavy Transportation Equipment": ("Industrials", "Capital Goods", "Machinery"),
    "Agricultural & Farm Machinery": ("Industrials", "Capital Goods", "Machinery"),
    "Industrial Machinery & Supplies & Components": ("Industrials", "Capital Goods", "Machinery"),
    "Heavy Electrical Equipment": ("Industrials", "Capital Goods", "Electrical Equipment"),
    "Trading Companies & Distributors": ("Industrials", "Capital Goods", "Trading Companies & Distributors"),
    "Commercial Printing": ("Industrials", "Commercial & Professional Services", "Commercial Printing"),
    "Professional Services": ("Industrials", "Commercial & Professional Services", "Professional Services"),
    "Research & Consulting Services": ("Industrials", "Commercial & Professional Services", "Professional Services"),
    "Environmental & Facilities Services": ("Industrials", "Commercial & Professional Services", "Environmental & Facilities Services"),
    "Office Services & Supplies": ("Industrials", "Commercial & Professional Services", "Office Services & Supplies"),
    "Diversified Support Services": ("Industrials", "Commercial & Professional Services", "Professional Services"),
    "Human Resource & Employment Services": ("Industrials", "Commercial & Professional Services", "Professional Services"),
    "Data Processing & Outsourced Services": ("Industrials", "Commercial & Professional Services", "Professional Services"),
    "Air Freight & Logistics": ("Industrials", "Transportation", "Air Freight & Logistics"),
    "Passenger Airlines": ("Industrials", "Transportation", "Airlines"),
    "Airlines": ("Industrials", "Transportation", "Airlines"),
    "Marine": ("Industrials", "Transportation", "Marine"),
    "Rail Transportation": ("Industrials", "Transportation", "Road & Rail"),
    "Cargo Ground Transportation": ("Industrials", "Transportation", "Road & Rail"),
    "Road & Rail": ("Industrials", "Transportation", "Road & Rail"),
    "Passenger Ground Transportation": ("Industrials", "Transportation", "Road & Rail"),
    "Transportation Infrastructure": ("Industrials", "Transportation", "Transportation Infrastructure"),
    
    # Information Technology
    "IT Consulting & Other Services": ("Information Technology", "Software & Services", "IT Consulting & Other Services"),
    "Internet Services & Infrastructure": ("Information Technology", "Software & Services", "Internet Services & Infrastructure"),
    "Data Processing & Outsourced Services": ("Information Technology", "Software & Services", "Data Processing & Outsourced Services"),
    "Application Software": ("Information Technology", "Software & Services", "Application Software"),
    "Systems Software": ("Information Technology", "Software & Services", "Systems Software"),
    "Home Entertainment Software": ("Information Technology", "Software & Services", "Home Entertainment Software"),
    "Communications Equipment": ("Information Technology", "Technology Hardware & Equipment", "Communications Equipment"),
    "Technology Hardware, Storage & Peripherals": ("Information Technology", "Technology Hardware & Equipment", "Technology Hardware, Storage & Peripherals"),
    "Electronic Equipment & Instruments": ("Information Technology", "Technology Hardware & Equipment", "Electronic Equipment & Instruments"),
    "Electronic Components": ("Information Technology", "Technology Hardware & Equipment", "Electronic Components"),
    "Electronic Manufacturing Services": ("Information Technology", "Technology Hardware & Equipment", "Electronic Manufacturing Services"),
    "Technology Distributors": ("Information Technology", "Technology Hardware & Equipment", "Technology Distributors"),
    "Semiconductor Materials & Equipment": ("Information Technology", "Semiconductors & Semiconductor Equipment", "Semiconductor Equipment"),
    "Semiconductor Equipment": ("Information Technology", "Semiconductors & Semiconductor Equipment", "Semiconductor Equipment"),
    "Semiconductors": ("Information Technology", "Semiconductors & Semiconductor Equipment", "Semiconductors"),
    
    # Materials
    "Commodity Chemicals": ("Materials", "Chemicals", "Commodity Chemicals"),
    "Diversified Chemicals": ("Materials", "Chemicals", "Diversified Chemicals"),
    "Fertilizers & Agricultural Chemicals": ("Materials", "Chemicals", "Fertilizers & Agricultural Chemicals"),
    "Industrial Gases": ("Materials", "Chemicals", "Industrial Gases"),
    "Specialty Chemicals": ("Materials", "Chemicals", "Specialty Chemicals"),
    "Construction Materials": ("Materials", "Construction Materials", "Construction Materials"),
    "Metal, Glass & Plastic Containers": ("Materials", "Containers & Packaging", "Metal & Glass Containers"),
    "Paper & Plastic Packaging Products & Materials": ("Materials", "Containers & Packaging", "Paper Packaging"),
    "Paper Packaging": ("Materials", "Containers & Packaging", "Paper Packaging"),
    "Aluminum": ("Materials", "Metals & Mining", "Aluminum"),
    "Diversified Metals & Mining": ("Materials", "Metals & Mining", "Diversified Metals & Mining"),
    "Copper": ("Materials", "Metals & Mining", "Copper"),
    "Gold": ("Materials", "Metals & Mining", "Gold"),
    "Precious Metals & Minerals": ("Materials", "Metals & Mining", "Precious Metals & Minerals"),
    "Silver": ("Materials", "Metals & Mining", "Silver"),
    "Steel": ("Materials", "Metals & Mining", "Steel"),
    "Forest Products": ("Materials", "Paper & Forest Products", "Forest Products"),
    "Paper Products": ("Materials", "Paper & Forest Products", "Paper Products"),
    
    # Real Estate
    "Diversified Real Estate Activities": ("Real Estate", "Real Estate Management & Development", "Diversified Real Estate Activities"),
    "Real Estate Operating Companies": ("Real Estate", "Real Estate Management & Development", "Real Estate Operating Companies"),
    "Real Estate Development": ("Real Estate", "Real Estate Management & Development", "Real Estate Development"),
    "Real Estate Services": ("Real Estate", "Real Estate Management & Development", "Real Estate Services"),
    "Office REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Multi-Family Residential REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Telecom Tower REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Data Center REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Self-Storage REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Retail REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Industrial REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Hotel & Resort REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Health Care REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Single-Family Residential REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Other Specialized REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    "Timber REITs": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Equity Real Estate Investment Trusts (REITs)"),
    
    # Utilities
    "Electric Utilities": ("Utilities", "Electric Utilities", "Electric Utilities"),
    "Gas Utilities": ("Utilities", "Gas Utilities", "Gas Utilities"),
    "Multi-Utilities": ("Utilities", "Multi-Utilities", "Multi-Utilities"),
    "Water Utilities": ("Utilities", "Water Utilities", "Water Utilities"),
    "Independent Power Producers & Energy Traders": ("Utilities", "Independent Power Producers & Energy Traders", "Independent Power and Renewable Electricity Producers"),
}


def load_industry_map(json_path):
    """Load the industry map JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_industry_map(json_path, data):
    """Save the industry map JSON file"""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def populate_from_csv(csv_path, json_path):
    """Populate the industry map with tickers from CSV"""
    # Load existing JSON
    industry_map = load_industry_map(json_path)
    
    # Track statistics
    stats = {
        'total_stocks': 0,
        'mapped_stocks': 0,
        'unmapped_stocks': 0,
        'unmapped_industries': set()
    }
    
    # Read CSV and populate
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total_stocks'] += 1
            
            symbol = row['Symbol']
            sub_industry = row['GICS Sub-Industry']
            
            # Look up the mapping
            if sub_industry in INDUSTRY_MAPPING:
                industry, sector, industry_group = INDUSTRY_MAPPING[sub_industry]
                
                # Navigate to the correct location in the JSON
                try:
                    target = industry_map['industries'][industry]['sectors'][sector]['industry_groups'][industry_group]
                    
                    # Add ticker if not already present
                    if symbol not in target:
                        target.append(symbol)
                        stats['mapped_stocks'] += 1
                except KeyError as e:
                    print(f"Warning: Path not found for {symbol} ({sub_industry}): {e}")
                    stats['unmapped_stocks'] += 1
                    stats['unmapped_industries'].add(sub_industry)
            else:
                print(f"Warning: No mapping for sub-industry '{sub_industry}' (Stock: {symbol})")
                stats['unmapped_stocks'] += 1
                stats['unmapped_industries'].add(sub_industry)
    
    # Save updated JSON
    save_industry_map(json_path, industry_map)
    
    # Print statistics
    print("\n" + "="*60)
    print("POPULATION COMPLETE")
    print("="*60)
    print(f"Total stocks processed: {stats['total_stocks']}")
    print(f"Successfully mapped: {stats['mapped_stocks']}")
    print(f"Unmapped: {stats['unmapped_stocks']}")
    
    if stats['unmapped_industries']:
        print(f"\nUnmapped sub-industries ({len(stats['unmapped_industries'])}):")
        for sub_ind in sorted(stats['unmapped_industries']):
            print(f"  - {sub_ind}")
    
    return stats


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "Valuation Comps - S&P500.csv"
    json_path = base_dir / "data" / "industry_map.json"
    
    print(f"CSV Path: {csv_path}")
    print(f"JSON Path: {json_path}")
    print(f"\nStarting population...\n")
    
    # Run the population
    populate_from_csv(csv_path, json_path)
    
    print("\nDone!")
