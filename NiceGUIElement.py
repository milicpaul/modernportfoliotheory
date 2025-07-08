from nicegui import ui
import psutil

class NiceGUIElement:
    def __init__(self):
        pass

    @staticmethod
    def DatePicker(text, val) -> ui.input:
        with ui.input(text) as date:
            with ui.menu().props('no-parent-event') as menu:
                with ui.date().bind_value(date):
                    with ui.row().classes('justify-end'):
                        ui.button('Close', on_click=menu.close).props('flat')
            with date.add_slot('append'):
                ui.icon('edit_calendar').on('click', menu.open).classes('cursor-pointer')
        date.value = val
        return date

    @staticmethod
    def FirstSplitterBefore(gui) -> ui.row:
        with ui.row() as firstRow:
            with ui.card():
                with ui.row():
                    gui.nbOfSimulation = ui.input(label='Portfolios simulation:', value='10')
                    gui.nbOfWeight = ui.input(label='Weights by simulation:', value='100000')
                    gui.kelly = ui.switch('Kelly', on_change=gui.ChangeValue, value=True)
                    gui.Robust = ui.switch('Robust', on_change=gui.ChangeValue)
                    gui.Sound = ui.switch("Sound", on_change=gui.ChangeValue)
                    gui.ShowLog = ui.switch("Show log")
                with ui.row():
                    mem = psutil.virtual_memory()
                    gui.label_total = ui.label(f"Tot: {mem.total / (1024 ** 3):.2f} Go")
                    gui.label_available = ui.label(f"Avalaible: {mem.available / (1024 ** 3):.2f} Go")
                    gui.label_used = ui.label(f"Used: {mem.used / (1024 ** 3):.2f} Go")
                    gui.label_percent = ui.label(f"RAM: {mem.percent}%")
                    gui.label_process = ui.label()
                    gui.queue_size = ui.label()
                    gui.temperature = ui.label()
            with ui.card():
                with ui.row():
                    gui.DateFrom = NiceGUIElement.DatePicker('Date From', '2018-01-01')
                    gui.DateTo = NiceGUIElement.DatePicker('Date To', '2022-12-31')
        return firstRow
    @staticmethod
    def ResultGrid(rowData, gridEvent):
        return ui.aggrid({
                'columnDefs': [
                    {"field": "returns"},
                    {"field": "volatility"},
                    {"field": "sharpe", "filter": 'agTextColumnFilter', 'floatingFilter': True},
                    {"field": "returns lowest vol"},
                    {"field": "lowest volatility"},
                    {"field": "sharpe lowest vol"},
                        {"field": "timestamp", "sort": "descending"},
                ],
                'rowData': rowData
                }).classes('max-h-50').style('width: 98%').on('cellClicked', lambda event: gridEvent)
    staticmethod
    def FinalPortfolio(assetsName,):
        return ui.aggrid({
            'columnDefs': [
                {"field": "Asset name"},
                {"field": "ISIN"},
                {"field": "Weight"}
            ],
            'rowData': assetsName}).classes('max-h-50').style('width: 98%')

    @staticmethod
    def FullAssets(fullAssetsList):
        return ui.aggrid({
            'columnDefs': [
                {"field": "Name", "filter": 'agTextColumnFilter', 'floatingFilter': False, 'checkboxSelection': True},
                {"field": "ISIN", "filter": 'agTextColumnFilter', 'floatingFilter': False},
                {"field": "Volatility"},
                {"field": "Sum positive"},
                {"field": "Sum negative"},
                {"field": "File name"},
            ],
            'rowData': fullAssetsList
        }).classes('max-h-300').style('width: 100%')

