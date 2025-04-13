from anyio import sleep
from nicegui import ui, events
import Portfolio
import  assets
import ParallelComputing
import asyncio
import traceback

loader = ui.label('Modern Portfolio Theory')
spinner = ui.spinner(size='lg', color='primary')
spinner.visible = False  # Caché tant que pas en traitement
with ui.splitter().style('width: 30%') as splitter:
    with splitter.before:
        nbOfSimulation = ui.input(label='Nb of portfolios simulation:', value='1')
    with splitter.after:
        nbOfWeight = ui.input(label='Nb of weight by simulation:', value='10000')
# On crée un spinner caché au début
rowData = []
with ui.splitter(horizontal=False, reverse=False, value=60,
                 on_change=lambda e: ui.notify(e.value)).style('width: 100%') as splitter:
    with splitter.before:
        aggrid = ui.aggrid({
            'columnDefs': [
                {"field": "returns"},
                {"field": "volatility"},
                {"field": "sharpe", "filter": 'agNumberColumnFilter', 'floatingFilter': True},
                {"field": "returns lowest vol"},
                {"field": "lowest volatility"},
                {"field": "sharpe lowest vol"},
            ],
            'rowData': rowData
        }).classes('max-h-40')
    with splitter.after:
        aggrid2 = ui.aggrid({
            'columnDefs': [
                {"field": "returns"},
                {"field": "volatility"},
                {"field": "sharpe", "filter": 'agNumberColumnFilter', 'floatingFilter': True},
                {"field": "returns lowest vol"},
                {"field": "lowest volatility"},
                {"field": "sharpe lowest vol"},
            ],
            'rowData': rowData
        }).classes('max-h-40')

a = ui.audio('https://cdn.pixabay.com/download/audio/2022/02/22/audio_d1718ab41b.mp3')
a.visible = False

def display_portfolio(bestPortfolio):
    rowData.append({"returns": round(bestPortfolio[2] * 100, 2), "volatility": round(bestPortfolio[3] * 100, 2),
     "sharpe": round(bestPortfolio[4], 2), "returns lowest vol": round(bestPortfolio[8] * 100, 2),
     "lowest volatility": round(bestPortfolio[7] * 100, 2), "sharpe lowest vol": round(bestPortfolio[9], 2)})
    aggrid.update()
    aggrid2.update()
    print(bestPortfolio)

async def Simulate(nbOfSimulation, nbOfWeight):
    portfolio = Portfolio.ModernPortfolioTheory(nbOfSimulation, 2, 4)
    portfolio.nbOfSimulatedWeights = nbOfWeight
    data, isin = portfolio.BuilHeterogeneousPortfolio(portfolio.portfolioStructure)
    showDensity = False
    isRandom = True
    localPortfolio = []
    bestPortfolios = ParallelComputing.Parallel.run_select_random_assets_parallel(portfolio, data, isin, nbOfSimulation, portfolio.portfolioStructure, showDensity, isRandom, localPortfolio)
    return bestPortfolios

async def Callback():
    try:
        btnSimulate.disable()
        loader.text = "Processing"
        await asyncio.sleep(0.3)
        spinner.visible = True
        bestPortfolio = await Simulate(int(nbOfSimulation.value), int(nbOfWeight.value))
        loader.text = "Terminated"
    except Exception as e:
        print("💥 ERREUR pendant la simulation :", e)
        traceback.print_exc()
        loader.text = "Erreur !"
    finally:
        await ui.context.client.connected()
        await asyncio.sleep(2)
        btnSimulate.enable()
        spinner.visible = False
        loader.text = "Simulation terminée"  # Forcer le rafraîchissement en changeant le texte
        display_portfolio(bestPortfolio)
        a.play()
    return bestPortfolio

def handle_theme_change(e: events.ValueChangeEventArguments):
    aggrid.classes(add='ag-theme-balham-dark' if e.value else 'ag-theme-balham',
                 remove='ag-theme-balham ag-theme-balham-dark')

ui.switch('Dark', on_change=handle_theme_change)

btnSimulate = ui.button('Simulate', on_click=Callback)
ui.run()
