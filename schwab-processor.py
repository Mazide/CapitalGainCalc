import pandas as pd
from datetime import datetime
import json
import sys
from pathlib import Path

class Logger:
    def __init__(self, filename='capital_gains_calculation.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class RSUProcessor:
    def __init__(self):
        self.tax_year_start = datetime(2023, 4, 6)
        self.tax_year_end = datetime(2024, 4, 5)
    
    def parse_date(self, date_str):
        """Parse date from US format, handling 'as of' suffix"""
        if pd.isna(date_str) or date_str == '':
            return None
        try:
            # Remove any "as of" suffix and trim
            clean_date = date_str.split(' as of ')[0].strip()
            return datetime.strptime(clean_date, '%m/%d/%Y')
        except ValueError as e:
            print(f"Error parsing date {date_str}: {e}")
            return None
    
    def clean_monetary_value(self, value: str) -> float:
        """Clean monetary values from symbols"""
        if pd.isna(value) or value == '':
            return 0.0
        return float(str(value).replace('$', '').replace(',', ''))
    
    def clean_quantity(self, value: str) -> float:
        """Clean share quantities"""
        if pd.isna(value) or value == '':
            return 0.0
        return float(str(value).replace(',', ''))
    
    def process_equity_data(self, file_path: str) -> pd.DataFrame:
        """Process RSU data"""
        print("\n=== Processing RSU Data ===")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from equity file")
        
        # Convert dates
        date_columns = ['Date', 'AwardDate']
        for col in date_columns:
            df[col] = df[col].apply(self.parse_date)
        
        # Clean numeric values
        df['FairMarketValuePrice'] = df['FairMarketValuePrice'].apply(self.clean_monetary_value)
        df['SalePrice'] = df['SalePrice'].apply(self.clean_monetary_value)
        df['Taxes'] = df['Taxes'].apply(self.clean_monetary_value)
        df['SharesSoldWithheldForTaxes'] = df['SharesSoldWithheldForTaxes'].apply(self.clean_quantity)
        df['NetSharesDeposited'] = df['NetSharesDeposited'].apply(self.clean_quantity)
        
        # Filter only records with shares
        df = df[df['NetSharesDeposited'] > 0].copy()
        
        # Sort by date for consistent processing
        df = df.sort_values('AwardDate')
        
        # Log processed vestings
        print("\nProcessed RSU Vestings:")
        for idx, row in df.iterrows():
            print(f"\nVesting {row['AwardDate'].strftime('%d/%m/%Y')}:")
            print(f"Received: {row['NetSharesDeposited']:.0f} shares at ${row['FairMarketValuePrice']:.2f}")
            if row['SharesSoldWithheldForTaxes'] > 0:
                print(f"Sold for tax: {row['SharesSoldWithheldForTaxes']:.0f} shares at ${row['SalePrice']:.2f}")
                print(f"Taxes paid: ${row['Taxes']:.2f}")
        
        return df
    
    def process_sales_data(self, file_path: str) -> pd.DataFrame:
        """Process sales data"""
        print("\n=== Processing Sales Data ===")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from sales file")
        
        # Convert dates
        df['Date'] = df['Date'].apply(self.parse_date)
        
        # Filter only sales
        df = df[df['Action'] == 'Sell'].copy()
        
        # Clean numeric values
        df['Price'] = df['Price'].apply(self.clean_monetary_value)
        df['Amount'] = df['Amount'].apply(self.clean_monetary_value)
        df['Quantity'] = df['Quantity'].apply(self.clean_quantity)
        
        # Sort by date for FIFO
        df = df.sort_values('Date')
        
        # Log all sales chronologically
        if len(df) > 0:
            print("\nAll sales in chronological order:")
            for idx, row in df.iterrows():
                if pd.notna(row['Date']):  # Only print if date is not NaT
                    print(f"\nSale {row['Date'].strftime('%d/%m/%Y')}:")
                    print(f"Sold: {row['Quantity']:.0f} shares at ${row['Price']:.2f}")
                    print(f"Total amount: ${row['Amount']:.2f}")
                else:
                    print(f"\nSale with invalid date:")
                    print(f"Sold: {row['Quantity']:.0f} shares at ${row['Price']:.2f}")
                    print(f"Total amount: ${row['Amount']:.2f}")
        
        return df

    def calculate_gains(self, equity_df: pd.DataFrame, sales_df: pd.DataFrame) -> dict:
        """Calculate capital gains"""
        print("\n=== Calculating Capital Gains ===")
        
        # 1. Automatic tax sales
        print("\nAutomatic Tax Sales:")
        tax_sales = []
        tax_sales_total = {
            'shares': 0,
            'proceeds': 0,
            'cost_basis': 0,
            'gain': 0,
            'taxes': 0
        }
        
        for idx, row in equity_df.iterrows():
            if row['SharesSoldWithheldForTaxes'] > 0:
                # Only process tax sales within the tax year
                if self.tax_year_start <= row['AwardDate'] <= self.tax_year_end:
                    shares = row['SharesSoldWithheldForTaxes']
                    cost_basis_per_share = row['FairMarketValuePrice']
                    sale_price = row['SalePrice']
                    
                    proceeds = shares * sale_price
                    cost_basis = shares * cost_basis_per_share
                    gain = proceeds - cost_basis
                    
                    print(f"\nVesting {row['AwardDate'].strftime('%d/%m/%Y')}:")
                    print(f"Received: {row['NetSharesDeposited']:.0f} shares at ${cost_basis_per_share:.2f}")
                    print(f"Sold for tax: {shares:.0f} shares at ${sale_price:.2f}")
                    print(f"Gain/Loss: ${gain:.2f}")
                    print(f"Taxes paid: ${row['Taxes']:.2f}")
                    
                    tax_sales.append({
                        'date': row['AwardDate'].strftime('%Y-%m-%d') if row['AwardDate'] else None,
                        'award_id': row['AwardId'],
                        'shares': shares,
                        'cost_basis_per_share': cost_basis_per_share,
                        'sale_price': sale_price,
                        'proceeds': proceeds,
                        'cost_basis': cost_basis,
                        'gain': gain,
                        'taxes_paid': row['Taxes']
                    })
                    
                    tax_sales_total['shares'] += shares
                    tax_sales_total['proceeds'] += proceeds
                    tax_sales_total['cost_basis'] += cost_basis
                    tax_sales_total['gain'] += gain
                    tax_sales_total['taxes'] += row['Taxes']
        
        # 2. Manual sales
        print("\nManual Sales:")
        manual_sales = []
        manual_sales_total = {
            'shares': 0,
            'proceeds': 0,
            'cost_basis': 0,
            'gain': 0
        }
        
        # Create FIFO vesting queue
        vesting_queue = []
        for idx, row in equity_df.iterrows():
            if row['NetSharesDeposited'] > 0:
                vest = {
                    'date': row['AwardDate'],
                    'award_id': row['AwardId'],
                    'shares': row['NetSharesDeposited'],
                    'cost_basis': row['FairMarketValuePrice']
                }
                vesting_queue.append(vest)
        
        # Sort vestings by date (ensuring FIFO)
        vesting_queue.sort(key=lambda x: x['date'])
        
        # Process all sales
        for idx, sale in sales_df.iterrows():
            # Only process sales within the tax year
            if self.tax_year_start <= sale['Date'] <= self.tax_year_end:
                print(f"\nSale {sale['Date'].strftime('%d/%m/%Y')}:")
                print(f"Selling: {sale['Quantity']:.0f} shares at ${sale['Price']:.2f}")
                
                remaining_shares = sale['Quantity']
                proceeds = sale['Amount']
                sale_price_per_share = sale['Price']
                cost_basis = 0
                lots_used = []
                
                while remaining_shares > 0 and vesting_queue and vesting_queue[0]['shares'] > 0:
                    vest = vesting_queue[0]
                    shares_from_lot = min(remaining_shares, vest['shares'])
                    
                    lot_cost_basis = shares_from_lot * vest['cost_basis']
                    lot_proceeds = shares_from_lot * sale_price_per_share
                    
                    print(f"  Used: {shares_from_lot:.0f} shares from {vest['date'].strftime('%d/%m/%Y')} vesting at ${vest['cost_basis']:.2f}")
                    
                    lots_used.append({
                        'award_id': vest['award_id'],
                        'vesting_date': vest['date'].strftime('%Y-%m-%d') if vest['date'] else None,
                        'shares': shares_from_lot,
                        'cost_basis_per_share': vest['cost_basis']
                    })
                    
                    cost_basis += lot_cost_basis
                    remaining_shares -= shares_from_lot
                    vest['shares'] -= shares_from_lot
                    
                    if vest['shares'] <= 0:
                        vesting_queue.pop(0)
                
                gain = proceeds - cost_basis
                print(f"  Gain/Loss: ${gain:.2f}")
                
                manual_sales.append({
                    'date': sale['Date'].strftime('%Y-%m-%d') if sale['Date'] else None,
                    'shares': sale['Quantity'],
                    'sale_price': sale_price_per_share,
                    'proceeds': proceeds,
                    'cost_basis': cost_basis,
                    'gain': gain,
                    'lots_used': lots_used
                })
                
                manual_sales_total['shares'] += sale['Quantity']
                manual_sales_total['proceeds'] += proceeds
                manual_sales_total['cost_basis'] += cost_basis
                manual_sales_total['gain'] += gain
        
        total_gain = tax_sales_total['gain'] + manual_sales_total['gain']
        
        return {
            'tax_sales': {
                'details': tax_sales,
                'summary': tax_sales_total
            },
            'manual_sales': {
                'details': manual_sales,
                'summary': manual_sales_total
            },
            'total_gain': total_gain
        }
    
    def print_report(self, results: dict):
        """Print final report"""
        print(f"\n=== Capital Gains Report {self.tax_year_start.year}-{self.tax_year_end.year} ===")
        print(f"Tax year: {self.tax_year_start.strftime('%d/%m/%Y')} - {self.tax_year_end.strftime('%d/%m/%Y')}")
        
        # Automatic tax sales
        tax = results['tax_sales']['summary']
        print("\nAutomatic Tax Sales:")
        print(f"Shares sold: {tax['shares']:.0f}")
        print(f"Proceeds: ${tax['proceeds']:,.2f}")
        print(f"Cost basis: ${tax['cost_basis']:,.2f}")
        print(f"Gain/Loss: ${tax['gain']:,.2f}")
        print(f"Taxes paid: ${tax['taxes']:,.2f}")
        
        # Manual sales
        manual = results['manual_sales']['summary']
        print("\nManual Sales:")
        print(f"Shares sold: {manual['shares']:.0f}")
        print(f"Proceeds: ${manual['proceeds']:,.2f}")
        print(f"Cost basis: ${manual['cost_basis']:,.2f}")
        print(f"Gain/Loss: ${manual['gain']:,.2f}")
        
        # Total results
        print("\nTotal Results:")
        print(f"Total Gain/Loss: ${results['total_gain']:,.2f}")
        
        # Save detailed report
        with open('reports/capital_gains_details.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nDetailed report saved to 'capital_gains_details.json'")

def main():
    try:
        # Create reports directory if it doesn't exist
        Path("reports").mkdir(exist_ok=True)
        
        # Initialize logger
        logger = Logger('reports/calculation_details.log')
        
        print("=== Starting Capital Gains Calculation ===")
        print(f"Date and time: {datetime.now()}\n")
        
        processor = RSUProcessor()
        
        # Process files
        equity_df = processor.process_equity_data('equity.csv')
        sales_df = processor.process_sales_data('history.csv')
        
        # Calculate gains
        results = processor.calculate_gains(equity_df, sales_df)
        
        # Print report
        processor.print_report(results)
        
        # Save results as JSON
        with open('reports/capital_gains_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nAll files saved in 'reports' directory:")
        print("1. calculation_details.log - Detailed calculation log")
        print("2. capital_gains_results.json - Results in JSON format")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
    finally:
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()