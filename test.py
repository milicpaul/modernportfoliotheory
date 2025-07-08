from nicegui import ui

# Création des tabs
with ui.tabs().classes('w-full') as tabs:
    tab1 = ui.tab('📈 Indices')
    tab2 = ui.tab('📊 Données')
    tab3 = ui.tab('⚙️ Paramètres')

# Contenu associé à chaque tab
with ui.tab_panels(tabs, value=tab1).classes('w-full'):
    with ui.tab_panel(tab1):
        ui.label('Voici les indices boursiers européens 📈')

    with ui.tab_panel(tab2):
        ui.label('Voici les données financières 📊')

    with ui.tab_panel(tab3):
        ui.label('Voici les options de configuration ⚙️')

ui.run()
