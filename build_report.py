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
    def __init__(self, date: str, action: str, symbol: str, quantity: str, amount: str, 
                 exchange_rates: pd.DataFrame, grant: Optional[EquityGrant] = None):
        transaction_date, vesting_date = parse_dates(date)
        self.date = pd.to_datetime(transaction_date)
        self.vesting_date = pd.to_datetime(vesting_date) if vesting_date else None
        self.action = action
        self.symbol = symbol
        self.quantity = int(float(str(quantity).replace(",", "")))
        
        # Конвертация суммы из USD в GBP
        usd_amount = float(str(amount).replace("$", "").replace(",", ""))
        self.exchange_rate = self._get_exchange_rate(exchange_rates)
        self.amount = usd_amount * (1 / self.exchange_rate)  # USD в GBP: умножаем на обратный курс
        
        self.grant = grant
        self.is_tax_sale = grant.shares_sold_for_taxes > 0 if grant else False
        
        # Расчет FMV в GBP
        self.fmv_total = (self.quantity * grant.fmv * (1 / self.exchange_rate)) if grant else 0  # USD в GBP
        self.gain = self.amount - self.fmv_total if grant else 0

    def _get_exchange_rate(self, exchange_rates: pd.DataFrame) -> float:
        """Get the exchange rate for the transaction date"""
        # Find the closest date that's not after our transaction date
        closest_date = exchange_rates.index[exchange_rates.index <= self.date].max()
        return exchange_rates.loc[closest_date, 'gbp_usd_rate']  # Changed column name

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
                f"FMV at Vest: £{(self.grant.fmv * (1 / self.exchange_rate)):,.2f}\n"
                f"Total FMV Value: £{self.fmv_total:,.2f}\n"
                f"Gain/Loss: £{self.gain:,.2f}\n"
                f"Award ID: {self.grant.award_id}\n"
                f"Tax Sale: {'Yes' if self.is_tax_sale else 'No'}\n"
                f"Tax Amount: £{(self.grant.taxes * (1 / self.exchange_rate)):,.2f}"
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

def write_year_report(year: int, realized_gains: List[RealizedGain], pool: Section104Pool, exchange_rates: pd.DataFrame, base_dir: str):
    """Write detailed report for a specific tax year in both TXT and PDF formats"""
    report_file = os.path.join(base_dir, f'tax_year_{year}_{year+1}_report.txt')
    
    def get_exchange_rate(date):
        """Get the exchange rate for the given date"""
        closest_date = exchange_rates.index[exchange_rates.index <= pd.Timestamp(date)].max()
        return exchange_rates.loc[closest_date, 'gbp_usd_rate']
    
    # Группируем операции по датам
    operations_by_date = defaultdict(list)
    acquisitions_by_date = defaultdict(list)
    for rg in realized_gains:
        if hasattr(rg, 'action') and rg.action == 'BUY':  # Используем только проверку на BUY
            acquisitions_by_date[rg.date.date()].append(rg)
        else:
            operations_by_date[rg.date.date()].append(rg)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        print(f"Capital Gains Tax Calculations for {year}-{year+1}", file=f)
        
        # Обрабатываем каждый день
        all_dates = sorted(set(operations_by_date.keys()) | set(acquisitions_by_date.keys()))
        for date in all_dates:
            print(f"\n{date.strftime('%d %B %Y')}", file=f)
            
            # Обрабатываем приобретения (acquisitions)
            if date in acquisitions_by_date:
                daily_acqs = acquisitions_by_date[date]
                exchange_rate = get_exchange_rate(date)
                
                for acq in daily_acqs:
                    cost_gbp = acq.cost / exchange_rate
                    price_per_unit = cost_gbp / acq.quantity
                    
                    print(f"Acquisition: {acq.quantity:,} units of {acq.symbol} "
                          f"at £{cost_gbp:,.2f} (£{price_per_unit:.2f}/unit)", file=f)
                    print(f"Number of units in the pool: {pool.total_shares:,}, "
                          f"new pool cost: £{pool.total_cost / exchange_rate:.2f} "
                          f"(£{pool.average_price / exchange_rate:.2f}/unit).", file=f)
                    print("\n")

            # Обрабатываем продажи (disposals)
            if date in operations_by_date:
                daily_ops = operations_by_date[date]
                exchange_rate = get_exchange_rate(date)
                
                for op in daily_ops:
                    proceeds_gbp = op.proceeds / exchange_rate
                    cost_gbp = op.cost / exchange_rate
                    gain_gbp = proceeds_gbp - cost_gbp
                    
                    # Определяем тип матчинга из атрибута realized gain
                    matching_type = getattr(op, 'action', 'SECTION_104')
                    
                    print(f"{matching_type}. Quantity: {op.quantity}, "
                          f"allowable cost: £{cost_gbp:.2f}, "
                          f"{'gain' if gain_gbp >= 0 else 'loss'}: £{abs(gain_gbp):.2f}", file=f)

        # Добавляем итоговое саммари в конец отчета
        summary = create_summary(realized_gains)
        print("\nOverall Summary:", file=f)
        print(f" Number of acquisitions: {summary['acquisitions']}", file=f)
        print(f" Number of disposals: {summary['disposals']}", file=f)
        print(f" Total disposal proceeds: £{summary['total_proceeds']:,.2f}", file=f)
        print(f" Total capital gain before loss: £{summary['total_gains']:,.2f}", file=f)
        print(f" Total capital loss: £{summary['total_losses']:,.2f}", file=f)
        print(f" Total capital gain after loss: £{summary['net_gain']:,.2f}", file=f)

class BedAndBreakfastTracker:
    def __init__(self):
        self.pending_sells = deque()
        self.realized_gains = []
        self.same_day_matches = {}  # Словарь для отслеживания same-day матчей
        self.bnb_matches = {}       # Словарь для отслеживания B&B матчей

    def add_sell(self, date: datetime, symbol: str, quantity: int, price: float):
        """Добавляет продажу в очередь ожидания"""
        self.pending_sells.append(Operation(date, 'SELL', symbol, quantity, price))

    def process_buy(self, buy_op: Operation) -> int:
        """Обрабатывает покупку, возвращает количество непокрытых акций"""
        remaining_qty = buy_op.quantity

        # Перебираем все продажи в 30-дневном окне
        i = 0
        while i < len(self.pending_sells):
            sell_op = self.pending_sells[i]
            
            if (buy_op.date <= sell_op.date + timedelta(days=30) and 
                buy_op.date > sell_op.date and 
                sell_op.symbol == buy_op.symbol):
                
                # Определяем количество акций для покрытия
                matched_qty = min(remaining_qty, sell_op.quantity)
                
                # Создаем realized gain с ценой покупки как базисом
                cost = matched_qty * buy_op.price
                proceeds = matched_qty * sell_op.price
                self.realized_gains.append(RealizedGain(
                    sell_op.date, sell_op.symbol, matched_qty, proceeds, cost
                ))

                # Обновляем количества
                remaining_qty -= matched_qty
                sell_op.quantity -= matched_qty
                
                # Удаляем полностью покрытые продажи
                if sell_op.quantity == 0:
                    self.pending_sells.remove(sell_op)
                else:
                    i += 1
                
                if remaining_qty == 0:
                    break
            else:
                i += 1

        return remaining_qty

    def process_expired_sells(self, current_date: datetime, pool: Section104Pool):
        """Обрабатывает просроченные продажи через Section 104 Pool"""
        while self.pending_sells:
            oldest_sell = self.pending_sells[0]
            if (current_date - oldest_sell.date).days > 30:
                avg_price = pool.remove_shares(oldest_sell.quantity)
                proceeds = oldest_sell.quantity * oldest_sell.price
                cost = oldest_sell.quantity * avg_price
                self.realized_gains.append(RealizedGain(
                    oldest_sell.date, oldest_sell.symbol, 
                    oldest_sell.quantity, proceeds, cost
                ))
                self.pending_sells.popleft()
            else:
                break

    def match_same_day(self, sell_op: Operation, buys: List[Operation]) -> Tuple[int, List[RealizedGain]]:
        """Сопоставляет продажи с покупками того же дня"""
        remaining_qty = sell_op.quantity
        matched_gains = []
        
        same_day_buys = [b for b in buys if b.date.date() == sell_op.date.date()]
        for buy in same_day_buys:
            if remaining_qty <= 0:
                break
                
            matched_qty = min(remaining_qty, buy.quantity)
            cost = matched_qty * buy.price
            proceeds = matched_qty * sell_op.price
            
            gain = RealizedGain(
                sell_op.date, 
                sell_op.symbol,
                matched_qty,
                proceeds,
                cost,
                'SAME_DAY'  # Добавляем тип матчинга
            )
            matched_gains.append(gain)
            
            # Сохраняем информацию о матчинге
            self.same_day_matches[sell_op] = (buy, matched_qty)
            
            remaining_qty -= matched_qty
            
        return remaining_qty, matched_gains

    def match_bed_and_breakfast(self, sell_op: Operation, buys: List[Operation], remaining_qty: int) -> Tuple[int, List[RealizedGain]]:
        """Сопоставляет продажи с покупками в 30-дневном окне"""
        matched_gains = []
        
        future_buys = [b for b in buys if 
                      b.date > sell_op.date and 
                      b.date <= sell_op.date + timedelta(days=30)]
                      
        for buy in future_buys:
            if remaining_qty <= 0:
                break
                
            matched_qty = min(remaining_qty, buy.quantity)
            cost = matched_qty * buy.price
            proceeds = matched_qty * sell_op.price
            
            gain = RealizedGain(
                sell_op.date,
                sell_op.symbol,
                matched_qty,
                proceeds,
                cost,
                'BED_AND_BREAKFAST'
            )
            matched_gains.append(gain)
            
            # Сохраняем информацию о матчинге
            self.bnb_matches[sell_op] = (buy, matched_qty)
            
            remaining_qty -= matched_qty
            
        return remaining_qty, matched_gains

    def process_transaction(self, transaction: Operation, pool: Section104Pool, all_transactions: List[Operation]) -> List[RealizedGain]:
        """Обрабатывает транзакцию согласно правилам HMRC"""
        if transaction.action == 'SELL':
            gains = []
            remaining_qty = transaction.quantity
            
            # 1. Same Day Rule
            remaining_qty, same_day_gains = self.match_same_day(transaction, all_transactions)
            gains.extend(same_day_gains)
            
            # 2. Bed and Breakfast Rule
            if remaining_qty > 0:
                remaining_qty, bnb_gains = self.match_bed_and_breakfast(transaction, all_transactions, remaining_qty)
                gains.extend(bnb_gains)
            
            # 3. Section 104 Pool
            if remaining_qty > 0:
                avg_price = pool.remove_shares(remaining_qty)
                proceeds = remaining_qty * transaction.price
                cost = remaining_qty * avg_price
                
                gains.append(RealizedGain(
                    transaction.date,
                    transaction.symbol,
                    remaining_qty,
                    proceeds,
                    cost,
                    'SECTION_104'
                ))
                
            return gains
            
        elif transaction.action == 'BUY':
            pool.add_shares(transaction.quantity, transaction.price)
            return []

def process_transactions_chronologically(history_file: str, equity_file: str, 
                                      exchange_rates_file: str, output_dir: str):
    """Process transactions and grants in chronological order."""
    equity_data = load_equity_data(equity_file)
    exchange_rates = load_exchange_rates(exchange_rates_file)
    transactions = load_transactions(history_file, equity_data, exchange_rates)
    
    operations = []
    all_realized_gains = []  # Добавляем список для всех операций
    
    # Создаем операции покупки из грантов
    for grant in equity_data:
        if grant.lapse_date:
            lapse_date = pd.to_datetime(grant.lapse_date)
            total_shares = grant.net_shares + grant.shares_sold_for_taxes
            closest_date = exchange_rates.index[exchange_rates.index <= lapse_date].max()
            exchange_rate = exchange_rates.loc[closest_date, 'gbp_usd_rate']
            
            # Создаем операцию покупки
            buy_op = Operation(
                date=lapse_date,
                action='BUY',
                symbol='RSU',
                quantity=total_shares,
                price=grant.fmv / exchange_rate
            )
            operations.append(buy_op)
            
            # Добавляем информацию о покупке в realized_gains
            all_realized_gains.append(RealizedGain(
                date=lapse_date,
                symbol='RSU',
                quantity=total_shares,
                proceeds=total_shares * (grant.fmv / exchange_rate),
                cost=total_shares * (grant.fmv / exchange_rate),
                action='BUY'  # Добавляем информацию о типе операции
            ))
    
    # Добавляем продажи
    for trans in transactions:
        if trans.action == 'Sell':
            operations.append(Operation(
                date=trans.date,
                action='SELL',
                symbol='RSU',
                quantity=-abs(trans.quantity),
                price=trans.amount / trans.quantity
            ))
    
    # Сортируем операции по дате
    operations.sort(key=lambda x: x.date)
    
    # Инициализируем трекеры
    pool = Section104Pool()
    bnb_tracker = BedAndBreakfastTracker()
    
    for op in operations:
        bnb_tracker.process_expired_sells(op.date, pool)
        
        if op.action == 'SELL':
            bnb_tracker.add_sell(op.date, op.symbol, abs(op.quantity), op.price)
        else:  # BUY
            remaining_qty = bnb_tracker.process_buy(op)
            if remaining_qty > 0:
                pool.add_shares(remaining_qty, op.price)
    
    # Обрабатываем оставшиеся продажи
    last_date = max(op.date for op in operations)
    bnb_tracker.process_expired_sells(last_date + timedelta(days=31), pool)
    
    # Добавляем все realized gains от B&B трекера
    all_realized_gains.extend(bnb_tracker.realized_gains)
    
    # Группируем по налоговым годам
    gains_by_year = defaultdict(list)
    for gain in all_realized_gains:  # Используем общий список all_realized_gains
        tax_year = get_uk_tax_year(gain.date)
        gains_by_year[tax_year].append(gain)
    
    # Создаем отчет для каждого налогового года
    for year, gains in gains_by_year.items():
        write_year_report(year, gains, pool, exchange_rates, output_dir)

def build_report(history_file: str, equity_file: str, exchange_rates_file: str):
    """Build a report of stock transactions by loading data from history and equity files."""
    base_dir = 'reports'
    os.makedirs(base_dir, exist_ok=True)
    
    process_transactions_chronologically(history_file, equity_file, exchange_rates_file, base_dir)
    
    print(f"Reports generated in directory: {base_dir}")

build_report('history.csv', 'equity.csv', 'exchange_rates.csv')