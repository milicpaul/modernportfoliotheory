from nicegui import ui
import Portfolio

nbOfSimulation = ui.input(label='Nb of portfolios simulation:', value=10)
nbOfWeight = ui.input(label='Nb of weight by simulation:', value=10000)

def notify():
    ui.notify(f'Started {nbOfSimulation.value} simulations!')

ui.button('Simulate', on_click=notify)

ui.run()
