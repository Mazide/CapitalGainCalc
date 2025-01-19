import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path

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
        
        # Расчет прибыли/убытков
        self.fmv_total = self.quantity * grant.fmv if grant else 0
        self.gain = self.amount - self.fmv_total if grant else 0

    def __str__(self):
        base_str = (
            f"Дата транзакции: {self.date.strftime('%m/%d/%Y')}\n"
            f"Дата вестинга: {self.vesting_date.strftime('%m/%d/%Y') if self.vesting_date else 'Н/Д'}\n"
            f"Количество: {self.quantity}\n"
            f"Сумма продажи: ${self.amount:,.2f}\n"
        )
        if self.grant:
            base_str += (
                f"FMV на дату вестинга: ${self.grant.fmv:,.2f}\n"
                f"Общая стоимость по FMV: ${self.fmv_total:,.2f}\n"
                f"Прибыль/убыток: ${self.gain:,.2f}\n"
                f"Award ID: {self.grant.award_id}\n"
                f"Налоговая продажа: {'Да' if self.is_tax_sale else 'Нет'}\n"
                f"Налог: ${self.grant.taxes:,.2f}"
            )
        return base_str

def parse_dates(date_str: str) -> Tuple[str, Optional[str]]:
    """Возвращает кортеж (дата_транзакции, дата_вестинга)"""
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
    
    # Сортируем DataFrame по дате, но сохраняем порядок строк для каждой даты
    for _, row in df.iterrows():
        # Пропускаем строки с операцией Journal
        print(row)
        if pd.notna(row['Action']) and row['Action'] == 'Journal':
            continue
            
        if pd.notna(row['Date']) and pd.notna(row['Action']) and row['Action'] == 'Lapse':  # Строка с датой и Lapse
            current_lapse_date = row['Date']
            current_lapse_quantity = int(float(str(row['Quantity']).replace(',', ''))) if pd.notna(row['Quantity']) else None
        elif pd.notna(row['AwardId']) and current_lapse_date and current_lapse_quantity:  # Строка с деталями гранта
            grant = EquityGrant(
                award_date=row['AwardDate'],
                award_id=row['AwardId'],
                fmv=float(str(row['FairMarketValuePrice']).replace('$', '')),
                sale_price=float(str(row['SalePrice']).replace('$', '')),
                shares_sold_for_taxes=int(float(str(row['SharesSoldWithheldForTaxes']).replace(',', ''))) if pd.notna(row['SharesSoldWithheldForTaxes']) else 0,
                net_shares=int(float(str(row['NetSharesDeposited']).replace(',', ''))) if pd.notna(row['NetSharesDeposited']) else 0,
                taxes=float(str(row['Taxes']).replace('$', '').replace(',', '')) if pd.notna(row['Taxes']) else 0.0,
                lapse_date=current_lapse_date,  # Добавляем информацию из предыдущей строки
                lapse_quantity=current_lapse_quantity  # Добавляем информацию из предыдущей строки
            )
            grants.append(grant)
            
            # Сбрасываем значения lapse после создания гранта
            current_lapse_date = None
            current_lapse_quantity = None
    
    return grants

def load_transactions(filename: str, equity_data: List[EquityGrant]) -> List[Transaction]:
    """Загружает транзакции из CSV файла и сопоставляет их с грантами акций."""
    df = pd.read_csv(filename)
    transactions = []
    
    # Сортируем DataFrame по дате
    df = df.sort_values(by='Date', ascending=True)
    
    # Группируем транзакции по дате для сопоставления с грантами
    date_quantity_map = {}
    
    for _, row in df.iterrows():
        if pd.isna(row['Date']) or pd.isna(row['Action']):
            continue
            
        date = row['Date']
        action = row['Action']
        
        # Обрабатываем только продажи и получение акций
        if action not in ['Sell', 'Stock Plan Activity']:
            continue
            
        quantity = row['Quantity'] if pd.notna(row['Quantity']) else 0
        amount = row['Amount'] if pd.notna(row['Amount']) else 0
        
        # Для Stock Plan Activity сохраняем количество акций
        if action == 'Stock Plan Activity':
            date_quantity_map[(date, str(quantity))] = True
            continue
            
        # Для продаж ищем соответствующий грант
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

def process_transactions_chronologically(history_file: str, equity_file: str, output_file):
    """
    Обрабатывает транзакции и гранты в хронологическом порядке.
    """
    equity_data = load_equity_data(equity_file)
    transactions = load_transactions(history_file, equity_data)
    
    # Для каждой транзакции ищем соответствующий грант
    for transaction in transactions:
        if transaction.action == 'Sell':
            transaction_date = transaction.date
            # Ищем соответствующий грант
            matching_grant = None
            for grant in equity_data:
                grant_date = pd.to_datetime(grant.lapse_date)
                # Проверяем совпадение даты (тот же день или предыдущий)
                dates_match = (grant_date == transaction_date) or (grant_date == transaction_date - pd.Timedelta(days=1))
                # Проверяем, соответствует ли количество проданных акций налоговой продаже
                if dates_match and grant.shares_sold_for_taxes == transaction.quantity:
                    transaction.grant = grant
                    transaction.is_tax_sale = True
                    break
    
    # Выводим транзакции
    for transaction in transactions:
        print(f"\nДата: {transaction.date.strftime('%m/%d/%Y')}", file=output_file)
        print("-" * 50, file=output_file)
        print("Тип: Транзакция", file=output_file)
        print(f"Количество акций: {transaction.quantity}", file=output_file)
        print(f"Налоговая продажа: {'Да' if transaction.is_tax_sale else 'Нет'}", file=output_file)
        print(f"Цена продажи: ${transaction.amount/transaction.quantity:,.2f}", file=output_file)
        if transaction.grant:
            print(f"Цена вестинга (FMV): ${transaction.grant.fmv:,.2f}", file=output_file)
        print("-" * 50, file=output_file)

def build_report(history_file: str, equity_file: str):
    """
    Строит отчет по транзакциям с акциями, загружая данные из файлов истории и грантов.
    """
    # Создаем директорию reports если она не существует
    os.makedirs('reports', exist_ok=True)
    
    # Генерируем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'reports/equity_report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        process_transactions_chronologically(history_file, equity_file, f)
        
    
    print(f"Отчет сохранен в файл: {report_file}")

build_report('history.csv', 'equity.csv')