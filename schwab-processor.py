import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
from collections import defaultdict

class EquityGrant:
    def __init__(self, award_date: str, award_id: str, fmv: float, sale_price: float, 
                 shares_sold_for_taxes: int, net_shares: int, taxes: float,
                 lapse_date: Optional[str] = None, lapse_quantity: Optional[int] = None):
        self.award_date = award_date
        self.award_id = award_id
        self.fmv = fmv
        self.sale_price = sale_price
        self.shares_sold_for_taxes = shares_sold_for_taxes
        self.net_shares = net_shares
        self.taxes = taxes
        self.lapse_date = lapse_date
        self.lapse_quantity = lapse_quantity

class Transaction:
    def __init__(self, date: str, action: str, symbol: str, quantity: str, amount: str, grant: Optional[EquityGrant] = None):
        transaction_date, vesting_date = parse_dates(date)
        self.date = pd.to_datetime(transaction_date)
        self.vesting_date = pd.to_datetime(vesting_date) if vesting_date else None
        self.action = action
        self.symbol = symbol
        self.quantity = int(str(quantity).replace(",", ""))
        self.amount = float(str(amount).replace("$", "").replace(",", ""))
        self.grant = grant
        self.is_tax_sale = grant.shares_sold_for_taxes > 0 if grant else False
        
        # Calculate profit/loss
        self.fmv_total = self.quantity * grant.fmv if grant else 0
        self.gain = self.amount - self.fmv_total if grant else 0

    def __str__(self):
        base_str = (
            f"Transaction Date: {self.date.strftime('%m/%d/%Y')}\n"
            f"Vesting Date: {self.vesting_date.strftime('%m/%d/%Y') if self.vesting_date else 'N/A'}\n"
            f"Quantity: {self.quantity}\n"
            f"Sale Amount: ${self.amount:,.2f}\n"
        )
        if self.grant:
            base_str += (
                f"FMV at Vest: ${self.grant.fmv:,.2f}\n"
                f"Total FMV Value: ${self.fmv_total:,.2f}\n"
                f"Gain/Loss: ${self.gain:,.2f}\n"
                f"Award ID: {self.grant.award_id}\n"
                f"Tax Sale: {'Yes' if self.is_tax_sale else 'No'}\n"
                f"Tax Amount: ${self.grant.taxes:,.2f}"
            )
        return base_str

def parse_dates(date_str: str) -> Tuple[str, Optional[str]]:
    """Returns tuple (transaction_date, vesting_date)"""
    if pd.isna(date_str):
        return None, None
    parts = str(date_str).split(' as of ')
    transaction_date = parts[0]
    vesting_date = parts[1] if len(parts) > 1 else None
    return transaction_date, vesting_date

def load_equity_data(filename: str) -> List[EquityGrant]:
    df = pd.read_csv(filename)
    grants = []
    
    current_lapse_date = None
    current_lapse_quantity = None
    
    for _, row in df.iterrows():
        # Skip Journal entries
        print(row)
        if pd.notna(row['Action']) and row['Action'] == 'Journal':
            continue
            
        if pd.notna(row['Date']) and pd.notna(row['Action']) and row['Action'] == 'Lapse':
            current_lapse_date = row['Date']
            current_lapse_quantity = int(float(str(row['Quantity']).replace(',', ''))) if pd.notna(row['Quantity']) else None
        elif pd.notna(row['AwardId']) and current_lapse_date and current_lapse_quantity:
            grant = EquityGrant(
                award_date=row['AwardDate'],
                award_id=row['AwardId'],
                fmv=float(str(row['FairMarketValuePrice']).replace('$', '')),
                sale_price=float(str(row['SalePrice']).replace('$', '')),
                shares_sold_for_taxes=int(float(str(row['SharesSoldWithheldForTaxes']).replace(',', ''))) if pd.notna(row['SharesSoldWithheldForTaxes']) else 0,
                net_shares=int(float(str(row['NetSharesDeposited']).replace(',', ''))) if pd.notna(row['NetSharesDeposited']) else 0,
                taxes=float(str(row['Taxes']).replace('$', '').replace(',', '')) if pd.notna(row['Taxes']) else 0.0,
                lapse_date=current_lapse_date,
                lapse_quantity=current_lapse_quantity
            )
            grants.append(grant)
            
            current_lapse_date = None
            current_lapse_quantity = None
    
    return grants

def load_transactions(filename: str, equity_data: List[EquityGrant]) -> List[Transaction]:
    """Load transactions from CSV file and match them with equity grants."""
    df = pd.read_csv(filename)
    transactions = []
    
    df = df.sort_values(by='Date', ascending=True)
    date_quantity_map = {}
    
    for _, row in df.iterrows():
        if pd.isna(row['Date']) or pd.isna(row['Action']):
            continue
            
        date = row['Date']
        action = row['Action']
        
        if action not in ['Sell', 'Stock Plan Activity']:
            continue
            
        quantity = row['Quantity'] if pd.notna(row['Quantity']) else 0
        amount = row['Amount'] if pd.notna(row['Amount']) else 0
        
        if action == 'Stock Plan Activity':
            date_quantity_map[(date, str(quantity))] = True
            continue
            
        grant = next((g for g in equity_data 
                     if g.lapse_date == date and str(g.net_shares + g.shares_sold_for_taxes) == str(quantity)), 
                    None)
        
        transaction = Transaction(
            date=date,
            action=action,
            symbol=row['Symbol'],
            quantity=quantity,
            amount=amount,
            grant=grant
        )
        transactions.append(transaction)
    
    return transactions

class Section104Pool:
    def __init__(self):
        self.total_shares = 0
        self.total_cost = 0.0
    
    @property
    def average_price(self):
        return self.total_cost / self.total_shares if self.total_shares else 0.0
    
    def add_shares(self, quantity, price):
        self.total_shares += quantity
        self.total_cost += quantity * price
    
    def remove_shares(self, quantity):
        avg = self.average_price
        self.total_shares -= quantity
        self.total_cost -= quantity * avg
        return avg

class Operation:
    def __init__(self, date: datetime, action: str, symbol: str, quantity: int, price: float):
        self.date = date
        self.action = action  # 'BUY' or 'SELL'
        self.symbol = symbol
        self.quantity = quantity
        self.price = price

class RealizedGain:
    def __init__(self, date: datetime, symbol: str, quantity: int, proceeds: float, cost: float):
        self.date = date
        self.symbol = symbol
        self.quantity = quantity
        self.proceeds = proceeds
        self.cost = cost
        self.gain = proceeds - cost

def get_uk_tax_year(date: datetime) -> int:
    year = date.year
    cutoff = datetime(year, 4, 5)
    return year - 1 if date <= cutoff else year

def write_year_report(year: int, realized_gains: List[RealizedGain], pool: Section104Pool, base_dir: str):
    """Write detailed report for a specific tax year"""
    year_dir = os.path.join(base_dir, f'tax_year_{year}_{year+1}')
    os.makedirs(year_dir, exist_ok=True)
    
    report_file = os.path.join(year_dir, f'capital_gains_report.txt')
    
    year_gains = [rg for rg in realized_gains if get_uk_tax_year(rg.date) == year]
    
    with open(report_file, 'w', encoding='utf-8') as f:
        print(f"CAPITAL GAINS REPORT FOR TAX YEAR {year}-{year+1}", file=f)
        print("=" * 80, file=f)
        
        # Add monthly breakdown
        monthly_summary = defaultdict(lambda: {'gains': 0.0, 'shares': 0, 'transactions': 0})
        for rg in year_gains:
            month = rg.date.strftime('%B %Y')
            monthly_summary[month]['gains'] += rg.gain
            monthly_summary[month]['shares'] += rg.quantity
            monthly_summary[month]['transactions'] += 1
        
        print("\nMONTHLY BREAKDOWN:", file=f)
        print("-" * 80, file=f)
        for month, data in sorted(monthly_summary.items()):
            print(f"{month}:", file=f)
            print(f"  Number of trades: {data['transactions']}", file=f)
            print(f"  Shares sold: {data['shares']:,}", file=f)
            print(f"  Profit/Loss: £{data['gains']:,.2f}", file=f)
            print("-" * 40, file=f)
        
        # Year summary statistics
        total_proceeds = sum(rg.proceeds for rg in year_gains)
        total_cost = sum(rg.cost for rg in year_gains)
        total_gain = sum(rg.gain for rg in year_gains)
        total_shares = sum(rg.quantity for rg in year_gains)
        
        print("\nYEAR SUMMARY:", file=f)
        print("-" * 80, file=f)
        print(f"Total Transactions: {len(year_gains)}", file=f)
        print(f"Total Shares Sold: {total_shares:,}", file=f)
        print(f"Total Proceeds: £{total_proceeds:,.2f}", file=f)
        print(f"Total Cost Basis: £{total_cost:,.2f}", file=f)
        print(f"Net Capital Gain: £{total_gain:,.2f}", file=f)
        if total_shares > 0:
            print(f"Average Gain per Share: £{(total_gain/total_shares):,.2f}", file=f)
        print("-" * 80, file=f)
        
        # Detailed transactions with improved formatting
        print("\nDETAILED TRANSACTIONS:", file=f)
        print("-" * 80, file=f)
        for rg in year_gains:
            print(f"Date: {rg.date.strftime('%d %B %Y')}", file=f)
            print(f"Shares Sold: {rg.quantity:,}", file=f)
            print("CALCULATION:", file=f)
            print(f"  Sale Price: £{(rg.proceeds/rg.quantity):,.2f} per share", file=f)
            print(f"  Cost Basis: £{(rg.cost/rg.quantity):,.2f} per share", file=f)
            print(f"  Profit per Share: £{((rg.proceeds-rg.cost)/rg.quantity):,.2f}", file=f)
            print("TOTALS:", file=f)
            print(f"  Proceeds: £{rg.proceeds:,.2f}", file=f)
            print(f"  Cost Basis: £{rg.cost:,.2f}", file=f)
            print(f"  Profit/Loss: £{rg.gain:,.2f}", file=f)
            print("-" * 80, file=f)

def process_transactions_chronologically(history_file: str, equity_file: str, output_dir: str):
    """Process transactions and grants in chronological order."""
    equity_data = load_equity_data(equity_file)
    transactions = load_transactions(history_file, equity_data)
    
    # Create operations list
    operations = []
    
    # Add vestings (BUY)
    for grant in equity_data:
        if grant.lapse_date:
            lapse_date = pd.to_datetime(grant.lapse_date)
            total_shares = grant.net_shares + grant.shares_sold_for_taxes
            operations.append(Operation(
                date=lapse_date,
                action='BUY',
                symbol='RSU',  # or use actual symbol
                quantity=total_shares,
                price=grant.fmv
            ))
    
    # Add sales (SELL)
    for trans in transactions:
        if trans.action == 'Sell':
            operations.append(Operation(
                date=trans.date,
                action='SELL',
                symbol='RSU',  # or use actual symbol
                quantity=-abs(trans.quantity),  # negative value for sales
                price=trans.amount / trans.quantity
            ))
    
    # Sort operations by date
    operations.sort(key=lambda x: x.date)
    
    # Process operations through Section 104 pool
    pool = Section104Pool()
    realized_gains = []
    
    for op in operations:
        if op.action == 'BUY':
            pool.add_shares(op.quantity, op.price)
        else:  # SELL
            quantity = abs(op.quantity)
            avg_price = pool.remove_shares(quantity)
            proceeds = quantity * op.price
            cost = quantity * avg_price
            realized_gains.append(RealizedGain(
                date=op.date,
                symbol=op.symbol,
                quantity=quantity,
                proceeds=proceeds,
                cost=cost
            ))
    
    # Group by tax year and create separate reports
    tax_years = set(get_uk_tax_year(rg.date) for rg in realized_gains)
    
    for year in tax_years:
        write_year_report(year, realized_gains, pool, output_dir)
    
    # Write summary report
    summary_file = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        print("OVERALL CAPITAL GAINS SUMMARY", file=f)
        print("=" * 80, file=f)
        
        for year in sorted(tax_years):
            year_gains = [rg for rg in realized_gains if get_uk_tax_year(rg.date) == year]
            total_gain = sum(rg.gain for rg in year_gains)
            print(f"\nTax Year {year}-{year+1}:", file=f)
            print(f"Number of Transactions: {len(year_gains)}", file=f)
            print(f"Total Capital Gain: £{total_gain:,.2f}", file=f)
            print("-" * 80, file=f)
        
        print("\nCURRENT SECTION 104 HOLDING:", file=f)
        print("-" * 80, file=f)
        print(f"Shares in Pool: {pool.total_shares:,}", file=f)
        print(f"Total Pool Cost: £{pool.total_cost:,.2f}", file=f)
        print(f"Average Share Cost: £{pool.average_price:,.2f}", file=f)

def build_report(history_file: str, equity_file: str):
    """Build a report of stock transactions by loading data from history and equity files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('reports', f'equity_report_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)
    
    process_transactions_chronologically(history_file, equity_file, base_dir)
    
    print(f"Reports generated in directory: {base_dir}")

build_report('history.csv', 'equity.csv')