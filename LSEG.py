import lseg.data as ld


ld.open_session()

# Récupérer des données pour une liste d'instruments
df = ld.get_data(
    universe=['AAPL.O', 'MSFT.O'],
    fields=['BID', 'ASK', 'TR.Revenue']
)

print(df)