import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
from collections import defaultdict, deque
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class EquityGrant:
    def __init__(self, award_date: str, award_id: str, fmv: float, sale_price: float, 
                 shares_sold_for_taxes: int, net_shares: int, taxes: float,
                 lapse_date: Optional[str] = None, lapse_quantity: Optional[int] = None,
                 symbol: Optional[str] = None):
        self.award_date = award_date
        self.award_id = award_id
        self.fmv = fmv
        self.sale_price = sale_price
        self.shares_sold_for_taxes = shares_sold_for_taxes
        self.net_shares = net_shares
        self.taxes = taxes
        self.lapse_date = lapse_date
        self.lapse_quantity = lapse_quantity
        self.symbol = symbol

class Transaction:
    def __init__(self, date: str, action: str, symbol: str, quantity: str, amount: str, 
                 exchange_rates: pd.DataFrame, grant: Optional[EquityGrant] = None):
        transaction_date, vesting_date = parse_dates(date)
        self.date = pd.to_datetime(transaction_date)
        self.vesting_date = pd.to_datetime(vesting_date) if vesting_date else None
        self.action = action
        self.symbol = symbol
        self.quantity = int(float(str(quantity).replace(",", "")))
        
        # Сумма уже в USD, конвертируем в GBP один раз
        usd_amount = float(str(amount).replace("$", "").replace(",", ""))
        self.exchange_rate = self._get_exchange_rate(exchange_rates)
        self.amount = usd_amount / self.exchange_rate  # USD в GBP
        
        self.grant = grant
        self.is_tax_sale = grant.shares_sold_for_taxes > 0 if grant else False
        
        # Расчет FMV в GBP (сейчас grant.fmv в USD)
        self.fmv_total = (self.quantity * grant.fmv / self.exchange_rate) if grant else 0
        self.gain = self.amount - self.fmv_total if grant else 0

    def _get_exchange_rate(self, exchange_rates: pd.DataFrame) -> float:
        """Get the exchange rate for the transaction date's month"""
        return get_exchange_rate_for_date(self.date, exchange_rates)

    def __str__(self):
        base_str = (
            f"Transaction Date: {self.date.strftime('%m/%d/%Y')}\n"
            f"Vesting Date: {self.vesting_date.strftime('%m/%d/%Y') if self.vesting_date else 'N/A'}\n"
            f"Quantity: {self.quantity}\n"
            f"Exchange Rate: {self.exchange_rate:.4f}\n"
            f"Sale Amount: £{self.amount:,.2f}\n"
        )
        if self.grant:
            base_str += (
                f"FMV at Vest: £{(self.grant.fmv / self.exchange_rate):,.2f}\n"
                f"Total FMV Value: £{self.fmv_total:,.2f}\n"
                f"Gain/Loss: £{self.gain:,.2f}\n"
                f"Award ID: {self.grant.award_id}\n"
                f"Tax Sale: {'Yes' if self.is_tax_sale else 'No'}\n"
                f"Tax Amount: £{(self.grant.taxes / self.exchange_rate):,.2f}"
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
    current_symbol = None
    
    for _, row in df.iterrows():
        # Skip Journal entries
        print(row)
        if pd.notna(row['Action']) and row['Action'] == 'Journal':
            continue
            
        if pd.notna(row['Date']) and pd.notna(row['Action']) and row['Action'] == 'Lapse':
            current_lapse_date = row['Date']
            current_lapse_quantity = int(float(str(row['Quantity']).replace(',', ''))) if pd.notna(row['Quantity']) else None
            current_symbol = row['Symbol'] if pd.notna(row['Symbol']) else None
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
                lapse_quantity=current_lapse_quantity,
                symbol=current_symbol
            )
            grants.append(grant)
            
            current_lapse_date = None
            current_lapse_quantity = None
            current_symbol = None
    
    return grants

def load_exchange_rates(filename: str) -> pd.DataFrame:
    """Load exchange rates from CSV file and return as DataFrame indexed by date"""
    rates_df = pd.read_csv(filename)
    rates_df['date'] = pd.to_datetime(rates_df['date'])
    rates_df.set_index('date', inplace=True)
    return rates_df

def load_transactions(filename: str, equity_data: List[EquityGrant], exchange_rates: pd.DataFrame) -> List[Transaction]:
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
            exchange_rates=exchange_rates,
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
        self.action = action  # 'BUY' или 'SELL'
        self.symbol = symbol
        self.quantity = quantity
        self.price = price

class RealizedGain:
    def __init__(self, date: datetime, symbol: str, quantity: int, proceeds: float, cost: float, action: str = 'SELL'):
        self.date = date
        self.symbol = symbol
        self.quantity = quantity
        self.proceeds = proceeds
        self.cost = cost
        self.gain = proceeds - cost
        self.action = action  # 'BUY' или 'SELL'

class MatchingTracker:
    def __init__(self):
        self.buys = deque()  # очередь всех покупок

    def add_buy(self, buy: Operation):
        self.buys.append(buy)

    def match_sale(self, sell: Operation) -> List[RealizedGain]:
        realized_gains = []
        remaining_quantity = sell.quantity
        
        # 1. Same-day matching
        same_day_buys = [b for b in self.buys if b.date.date() == sell.date.date()]
        remaining_quantity, same_day_gains = self._match_buys(
            same_day_buys, sell, remaining_quantity, 'SAME_DAY'
        )
        realized_gains.extend(same_day_gains)

        # 2. Bed and Breakfast matching
        if remaining_quantity > 0:
            cutoff_date = sell.date - timedelta(days=30)
            bnb_buys = [b for b in self.buys 
                       if cutoff_date < b.date < sell.date]
            remaining_quantity, bnb_gains = self._match_buys(
                bnb_buys, sell, remaining_quantity, 'BED_AND_BREAKFAST'
            )
            realized_gains.extend(bnb_gains)

        # 3. Section 104 для оставшихся акций
        if remaining_quantity > 0:
            # Рассчитываем среднюю цену по оставшимся покупкам
            remaining_buys = [b for b in self.buys if b.quantity > 0]
            total_shares = sum(b.quantity for b in remaining_buys)
            total_cost = sum(b.quantity * b.price for b in remaining_buys)
            avg_price = total_cost / total_shares if total_shares else 0.0

            # Уменьшаем количество акций пропорционально
            for buy in remaining_buys:
                reduction = (buy.quantity * remaining_quantity) / total_shares
                buy.quantity -= reduction

            realized_gains.append(RealizedGain(
                date=sell.date,
                symbol=sell.symbol,
                quantity=remaining_quantity,
                proceeds=remaining_quantity * sell.price,
                cost=remaining_quantity * avg_price,
                action='SECTION_104'
            ))

        # Очищаем использованные покупки
        self.buys = deque(b for b in self.buys if b.quantity > 0)
        
        return realized_gains

    def _match_buys(self, buys: List[Operation], sell: Operation, 
                    quantity: int, match_type: str) -> Tuple[int, List[RealizedGain]]:
        realized_gains = []
        remaining = quantity

        for buy in buys:
            if remaining <= 0:
                break

            match_quantity = min(remaining, buy.quantity)
            
            realized_gains.append(RealizedGain(
                date=sell.date,
                symbol=sell.symbol,
                quantity=match_quantity,
                proceeds=match_quantity * sell.price,
                cost=match_quantity * buy.price,
                action=match_type
            ))

            remaining -= match_quantity
            buy.quantity -= match_quantity
            
        return remaining, realized_gains

def get_uk_tax_year(date: datetime) -> int:
    year = date.year
    cutoff = datetime(year, 4, 5)
    return year - 1 if date <= cutoff else year

def create_summary(realized_gains: List[RealizedGain]) -> dict:
    """Создает итоговое саммари по всем операциям"""
    acquisitions = set()
    disposals = set()
    total_proceeds = 0.0
    total_gains = 0.0
    total_losses = 0.0
    
    for gain in realized_gains:
        if hasattr(gain, 'action') and gain.action == 'BUY':
            acquisitions.add(gain.date)
        else:
            disposals.add(gain.date)
            total_proceeds += gain.proceeds
            if gain.gain > 0:
                total_gains += gain.gain
            else:
                total_losses += abs(gain.gain)
    
    return {
        'acquisitions': len(acquisitions),
        'disposals': len(disposals),
        'total_proceeds': total_proceeds,
        'total_gains': total_gains,
        'total_losses': total_losses,
        'net_gain': total_gains - total_losses
    }

def get_exchange_rate_for_date(date: datetime, exchange_rates: pd.DataFrame) -> float:
    """Get the exchange rate for the given date's month"""
    month_rates = exchange_rates[
        (exchange_rates.index.year == date.year) & 
        (exchange_rates.index.month == date.month)
    ]
    
    if month_rates.empty:
        raise ValueError(
            f"Не найден курс обмена для {date.strftime('%B %Y')}. "
            f"Убедитесь, что файл exchange_rates.csv содержит данные для этого периода."
        )
        
    return month_rates.iloc[0]['gbp_usd_rate']

def process_transactions_chronologically(history_file: str, equity_file: str, 
                                      exchange_rates_file: str) -> List[Operation]:
    """Обрабатывает транзакции и гранты хронологически и возвращает список операций."""
    exchange_rates = load_exchange_rates(exchange_rates_file)
    equity_data = load_equity_data(equity_file)
    transactions = load_transactions(history_file, equity_data, exchange_rates)
    
    operations = []
    
    # Создаем операции покупки из грантов
    for grant in equity_data:
        if grant.lapse_date:
            lapse_date = pd.to_datetime(grant.lapse_date)
            total_shares = grant.net_shares + grant.shares_sold_for_taxes
            exchange_rate = get_exchange_rate_for_date(lapse_date, exchange_rates)
            
            operations.append(Operation(
                date=lapse_date,
                action='BUY',
                symbol=grant.symbol or 'RSU',
                quantity=total_shares,
                price=grant.fmv / exchange_rate
            ))
    
    # Добавляем продажи
    for trans in transactions:
        if trans.action == 'Sell':
            operations.append(Operation(
                date=trans.date,
                action='SELL',
                symbol=trans.symbol,
                quantity=trans.quantity,
                price=trans.amount / trans.quantity
            ))
    
    # Сортируем операции по дате
    operations.sort(key=lambda x: x.date)
    return operations

def write_year_report(year: int, operations: List[Operation], bnb_tracker: MatchingTracker, 
                     exchange_rates: pd.DataFrame, base_dir: str) -> MatchingTracker:
    """Создает отчет для конкретного налогового года и возвращает обновленное состояние трекера."""
    txt_file = os.path.join(base_dir, f'tax_year_{year}_{year+1}_report.txt')
    pdf_file = os.path.join(base_dir, f'tax_year_{year}_{year+1}_report.pdf')
    
    realized_gains = []
    
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    content = []
    content.append(f"Capital Gains Tax Calculations for {year}-{year+1}\n")
    elements.append(Paragraph(content[-1], styles['Title']))
    elements.append(Spacer(1, 20))
    
    operations_by_date = defaultdict(list)
    for op in operations:
        operations_by_date[op.date.date()].append(op)
    
    for date in sorted(operations_by_date.keys()):
        daily_ops = operations_by_date[date]
        
        buys = [op for op in daily_ops if op.action == 'BUY']
        sells = [op for op in daily_ops if op.action == 'SELL']

        # Обработка операций
        for buy in buys:
            bnb_tracker.add_buy(buy)
            realized_gains.append(RealizedGain(
                buy.date, buy.symbol, buy.quantity, 0.0,
                buy.quantity * buy.price, 'Acquisition'
            ))
        
        for sell in sells:
            matching_gains = bnb_tracker.match_sale(sell)
            realized_gains.extend(matching_gains)
        
        # Форматирование вывода
        formatted_date = date.strftime("%d %B %Y")
        content.append(f"\n{formatted_date}")
        
        elements.append(Paragraph(f"Date: {formatted_date}", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Группировка операций по символу и цене
        buys_by_symbol_price = defaultdict(lambda: defaultdict(int))
        sells_by_symbol_price = defaultdict(lambda: defaultdict(int))
        
        for op in buys:
            buys_by_symbol_price[op.symbol][op.price] += op.quantity
            
        for op in sells:
            sells_by_symbol_price[op.symbol][op.price] += op.quantity
        
        # Вывод сгруппированных операций
        for symbol, price_qty in buys_by_symbol_price.items():
            for price, quantity in price_qty.items():
                line = f"Acquisition: {quantity} shares of {symbol} at £{price:.2f}"
                content.append(line)
                elements.append(Paragraph(line, styles['Normal']))
            
        for symbol, price_qty in sells_by_symbol_price.items():
            for price, quantity in price_qty.items():
                line = f"Sell: {quantity} shares of {symbol} at £{price:.2f}"
                content.append(line)
                elements.append(Paragraph(line, styles['Normal']))
        
        daily_section104_gains = [g for g in realized_gains 
                                if g.date.date() == date and g.action == 'SECTION_104']
        
        if daily_section104_gains:
            remaining_buys = [b for b in bnb_tracker.buys if b.quantity > 0]
            total_shares = sum(b.quantity for b in remaining_buys)
            total_cost = sum(b.quantity * b.price for b in remaining_buys)
            cost_basis_per_unit = total_cost / total_shares if total_shares else 0.0
            
            content.append("\nSection 104 Pool Status:")
            content.append(f"Total shares: {total_shares}")
            content.append(f"Cost basis per unit: £{cost_basis_per_unit:.2f}")
            content.append(f"Total cost: £{total_cost:.2f}")
            
            elements.append(Spacer(1, 15))
            for line in content[-4:]:
                elements.append(Paragraph(line, styles['Normal']))
        
        daily_gains = [g for g in realized_gains if g.date.date() == date and g.action != 'Acquisition']
        if daily_gains:
            content.append("\nRealized Gains/Losses:")
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Realized Gains/Losses:", styles['Normal']))
            
            for gain in daily_gains:
                cost_basis_per_unit = gain.cost / gain.quantity if gain.quantity else 0
                action = "Bed and Breakfast" if gain.action == 'BED_AND_BREAKFAST' else gain.action
                line = (f"{action}: {gain.quantity} shares, "
                       f"proceeds: £{gain.proceeds:.2f}, "
                       f"cost basis per unit: £{cost_basis_per_unit:.2f}, "
                       f"cost: £{gain.cost:.2f}, "
                       f"gain/loss: £{gain.gain:.2f}")
                content.append(line)
                elements.append(Paragraph(line, styles['Normal']))
        
        elements.append(Spacer(1, 20))
    
    summary = create_summary(realized_gains)
    
    summary_lines = [
        "\n" + "=" * 50,
        "YEARLY SUMMARY",
        "=" * 50,
        f"Number of acquisitions: {summary['acquisitions']}",
        f"Number of disposals: {summary['disposals']}",
        f"Total proceeds: £{summary['total_proceeds']:.2f}",
        f"Total gains: £{summary['total_gains']:.2f}",
        f"Total losses: £{summary['total_losses']:.2f}",
        f"Net gain/loss: £{summary['net_gain']:.2f}",
        "=" * 50 + "\n"
    ]
    
    content.extend(summary_lines)
    
    elements.append(Spacer(1, 30))
    for line in summary_lines:
        elements.append(Paragraph(line, styles['Normal']))
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    doc.build(elements)
    
    return bnb_tracker

def build_report(history_file: str, equity_file: str, exchange_rates_file: str):
    """Создает отчет по транзакциям акций."""
    base_dir = 'reports'
    os.makedirs(base_dir, exist_ok=True)
    
    exchange_rates = load_exchange_rates(exchange_rates_file)
    operations = process_transactions_chronologically(history_file, equity_file, exchange_rates_file)
    
    # Группируем операции по налоговым годам
    ops_by_year = defaultdict(list)
    for op in operations:
        tax_year = get_uk_tax_year(op.date)
        ops_by_year[tax_year].append(op)
    
    # Создаем отчеты по годам
    bnb_tracker = MatchingTracker()  # Создаем один трекер для всех лет
    for year in sorted(ops_by_year.keys()):
        bnb_tracker = write_year_report(year, ops_by_year[year], bnb_tracker, exchange_rates, base_dir)
    
    print(f"Отчеты сгенерированы в директории: {base_dir}")


build_report('history.csv', 'equity.csv', 'exchange_rates.csv')