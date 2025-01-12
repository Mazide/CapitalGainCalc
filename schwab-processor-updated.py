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
        """Парсинг даты из американского формата"""
        if pd.isna(date_str) or date_str == '':
            return None
        try:
            # Преобразование из формата MM/DD/YYYY
            return datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError as e:
            print(f"Error parsing date {date_str}: {e}")
            return None
    
    def clean_monetary_value(self, value: str) -> float:
        """Очистка денежных значений от символов"""
        if pd.isna(value) or value == '':
            return 0.0
        return float(str(value).replace('$', '').replace(',', ''))
    
    def clean_quantity(self, value: str) -> float:
        """Очистка количества акций"""
        if pd.isna(value) or value == '':
            return 0.0
        return float(str(value).replace(',', ''))
    
    def process_equity_data(self, file_path: str) -> pd.DataFrame:
        """Обработка файла с RSU"""
        print("\nProcessing equity data...")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from equity file")
        
        # Конвертация дат с новой функцией парсинга
        date_columns = ['Date', 'AwardDate']
        for col in date_columns:
            df[col] = df[col].apply(self.parse_date)
            print(f"Converted {col} dates")
        
        # Очистка числовых значений
        df['FairMarketValuePrice'] = df['FairMarketValuePrice'].apply(self.clean_monetary_value)
        df['SalePrice'] = df['SalePrice'].apply(self.clean_monetary_value)
        df['Taxes'] = df['Taxes'].apply(self.clean_monetary_value)
        df['SharesSoldWithheldForTaxes'] = df['SharesSoldWithheldForTaxes'].apply(self.clean_quantity)
        df['NetSharesDeposited'] = df['NetSharesDeposited'].apply(self.clean_quantity)
        
        # Фильтруем только записи с акциями
        df = df[df['NetSharesDeposited'] > 0].copy()
        print(f"Found {len(df)} records with shares deposited")
        
        return df
    
    def process_sales_data(self, file_path: str) -> pd.DataFrame:
        """Обработка файла с продажами"""
        print("\nProcessing sales data...")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from sales file")
        
        # Конвертация дат с новой функцией парсинга
        df['Date'] = df['Date'].apply(self.parse_date)
        print("Converted dates")
        
        # Фильтрация только продаж
        df = df[df['Action'] == 'Sell'].copy()
        print(f"Found {len(df)} sell transactions")
        
        # Очистка числовых значений
        df['Price'] = df['Price'].apply(self.clean_monetary_value)
        df['Amount'] = df['Amount'].apply(self.clean_monetary_value)
        df['Quantity'] = df['Quantity'].apply(self.clean_quantity)
        
        # Фильтрация по налоговому году
        mask = (df['Date'] >= self.tax_year_start) & (df['Date'] <= self.tax_year_end)
        df = df[mask].copy()
        print(f"Found {len(df)} sales in tax year")
        
        # Выводим найденные продажи для проверки
        if len(df) > 0:
            print("\nSales found in tax year:")
            for idx, row in df.iterrows():
                print(f"\nSale {idx + 1}:")
                print(f"Date: {row['Date']}")
                print(f"Quantity: {row['Quantity']}")
                print(f"Price: ${row['Price']}")
                print(f"Amount: ${row['Amount']}")
        
        return df
    
    def calculate_gains(self, equity_df: pd.DataFrame, sales_df: pd.DataFrame) -> dict:
        """Расчет прибыли"""
        print("\nCalculating gains...")
        
        # 1. Автоматические продажи для налогов
        print("\nProcessing automatic tax sales:")
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
                print(f"\nProcessing tax sale {idx + 1}:")
                shares = row['SharesSoldWithheldForTaxes']
                cost_basis_per_share = row['FairMarketValuePrice']
                sale_price = row['SalePrice']
                
                proceeds = shares * sale_price
                cost_basis = shares * cost_basis_per_share
                gain = proceeds - cost_basis
                
                print(f"Award ID: {row['AwardId']}")
                print(f"Shares sold for tax: {shares}")
                print(f"Cost basis per share: ${cost_basis_per_share:.4f}")
                print(f"Sale price per share: ${sale_price:.4f}")
                print(f"Total proceeds: ${proceeds:.2f}")
                print(f"Total cost basis: ${cost_basis:.2f}")
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
        
        print(f"\nProcessed {len(tax_sales)} automatic tax sales")
        print("\nTax sales summary:")
        print(json.dumps(tax_sales_total, indent=2))
        
        # 2. Ручные продажи
        print("\nProcessing manual sales:")
        
        # Создаем FIFO очередь вестингов
        vesting_queue = []
        print("\nBuilding vesting queue:")
        for idx, row in equity_df.iterrows():
            if row['NetSharesDeposited'] > 0:
                vest = {
                    'date': row['AwardDate'],
                    'award_id': row['AwardId'],
                    'shares': row['NetSharesDeposited'],
                    'cost_basis': row['FairMarketValuePrice']
                }
                print(f"\nVesting {idx + 1}:")
                print(f"Award ID: {vest['award_id']}")
                print(f"Date: {vest['date']}")
                print(f"Shares: {vest['shares']}")
                print(f"Cost basis: ${vest['cost_basis']:.4f}")
                vesting_queue.append(vest)
        
        # Сортируем вестинги и продажи по дате
        vesting_queue.sort(key=lambda x: x['date'])
        sales_df = sales_df.sort_values('Date')
        
        manual_sales = []
        manual_sales_total = {
            'shares': 0,
            'proceeds': 0,
            'cost_basis': 0,
            'gain': 0
        }
        
        print(f"\nProcessing {len(sales_df)} manual sales...")
        for idx, sale in sales_df.iterrows():
            print(f"\nProcessing sale {idx + 1}:")
            print(f"Date: {sale['Date']}")
            print(f"Quantity: {sale['Quantity']}")
            print(f"Price: ${sale['Price']:.4f}")
            
            remaining_shares = sale['Quantity']
            proceeds = sale['Amount']
            sale_price_per_share = sale['Price']
            cost_basis = 0
            lots_used = []
            
            print("\nMatching with vesting lots:")
            while remaining_shares > 0 and vesting_queue and vesting_queue[0]['shares'] > 0:
                vest = vesting_queue[0]
                shares_from_lot = min(remaining_shares, vest['shares'])
                
                lot_cost_basis = shares_from_lot * vest['cost_basis']
                lot_proceeds = shares_from_lot * sale_price_per_share
                
                print(f"\nUsing lot from {vest['award_id']}:")
                print(f"Shares from lot: {shares_from_lot}")
                print(f"Cost basis: ${lot_cost_basis:.2f}")
                print(f"Proceeds: ${lot_proceeds:.2f}")
                
                lots_used.append({
                    'award_id': vest['award_id'],
                    'vesting_date': vest['date'].strftime('%Y-%m-%d') if vest['date'] else None,
                    'shares': shares_from_lot,
                    'cost_basis_per_share': vest['cost_basis']
                })
                
                cost_basis += lot_cost_basis
                remaining_shares -= shares_from_lot
                vest['shares'] -= shares_from_lot
                
                print(f"Remaining shares in lot: {vest['shares']}")
                print(f"Remaining shares to match: {remaining_shares}")
                
                if vest['shares'] <= 0:
                    print(f"Lot {vest['award_id']} depleted, removing from queue")
                    vesting_queue.pop(0)
            
            gain = proceeds - cost_basis
            print(f"\nSale summary:")
            print(f"Total proceeds: ${proceeds:.2f}")
            print(f"Total cost basis: ${cost_basis:.2f}")
            print(f"Gain/Loss: ${gain:.2f}")
            
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
        
        print("\nManual sales summary:")
        print(json.dumps(manual_sales_total, indent=2))
        
        total_gain = tax_sales_total['gain'] + manual_sales_total['gain']
        print(f"\nTotal capital gains: ${total_gain:.2f}")
        
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
        """Вывод отчета"""
        print("\n=== Capital Gains Report 2023-2024 ===\n")
        
        # Автоматические продажи
        tax = results['tax_sales']['summary']
        print("Automatic Tax Sales:")
        print(f"Shares sold: {tax['shares']:.0f}")
        print(f"Proceeds: ${tax['proceeds']:,.2f}")
        print(f"Cost basis: ${tax['cost_basis']:,.2f}")
        print(f"Gain/Loss: ${tax['gain']:,.2f}")
        print(f"Taxes paid: ${tax['taxes']:,.2f}")
        
        # Ручные продажи
        manual = results['manual_sales']['summary']
        print("\nManual Sales:")
        print(f"Shares sold: {manual['shares']:.0f}")
        print(f"Proceeds: ${manual['proceeds']:,.2f}")
        print(f"Cost basis: ${manual['cost_basis']:,.2f}")
        print(f"Gain/Loss: ${manual['gain']:,.2f}")
        
        # Общий итог
        print("\nTotal Results:")
        print(f"Total Gain/Loss: ${results['total_gain']:,.2f}")
        
        # Сохраняем детальный отчет
        with open('reports/capital_gains_details.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nDetailed report saved to 'capital_gains_details.json'")

def main():
    try:
        # Создаем папку для отчетов если её нет
        Path("reports").mkdir(exist_ok=True)
        
        # Инициализируем логгер
        logger = Logger('reports/calculation_details.log')
        
        print("=== Starting Capital Gains Calculation ===")
        print(f"Date and time: {datetime.now()}\n")
        
        processor = RSUProcessor()
        
        # Обработка файлов
        equity_df = processor.process_equity_data('equity.csv')
        sales_df = processor.process_sales_data('history.csv')
        
        # Расчет прибыли
        results = processor.calculate_gains(equity_df, sales_df)
        
        # Вывод отчета
        processor.print_report(results)
        
        # Сохраняем JSON с результатами
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