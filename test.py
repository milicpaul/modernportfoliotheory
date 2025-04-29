from nicegui import ui, run
import asyncio
import pynvml

# Initialisation NVML une seule fois
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

# Création de jauges NiceGUI
gpu_util_gauge = ui.linear_progress(color='green', show_value=True)
mem_util_gauge = ui.linear_progress(color='orange', show_value=True)
info_label = ui.label('Initialisation...')

async def update_gpu_info():
    while True:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        mem_used = mem.used / 1024**2
        mem_total = mem.total / 1024**2
        mem_percent = mem_used / mem_total

        gpu_util = util.gpu / 100
        mem_util = util.memory / 100

        # MAJ UI
        gpu_util_gauge.set_value(gpu_util)
        mem_util_gauge.set_value(mem_util)
        info_label.set_text(
            f'GPU: {util.gpu}% | VRAM: {mem_used:.1f}/{mem_total:.1f} Mo ({mem_percent*100:.1f}%)'
        )

        await asyncio.sleep(1)

@ui.page('/')
async def main():
    ui.label('🎮 Utilisation GPU (temps réel)').classes('text-xl font-bold mb-4')
    await update_gpu_info()

ui.run()
