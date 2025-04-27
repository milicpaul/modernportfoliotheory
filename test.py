from multiprocessing import Process, Event, Queue, set_start_method
import multiprocessing
import time
import random
from nicegui import ui
import threading

def worker(name: str, alert_event: Event, queue: Queue, i: int):
    while True:
        time.sleep(random.uniform(3, 6))  # simulation d’un travail
        print(f"[{name}] J'ai terminé ou rencontré un souci.")
        queue.put(i)
        alert_event.set()

event = Event()
queue = Queue()

def start_workers(event, queue):
    workers = [Process(target=worker, args=(f"Worker-{i}", event, queue, i)) for i in range(10)]
    for w in workers:
        w.start()
    return workers

def update_uis(label):
    while True:
        print("[Main] J'attends qu’un worker me signale quelque chose...")
        event.wait()
        try:
            process_id = queue.get_nowait()
            label.text = f"C'est le process {process_id} qui te parle"
            label.update()
            print(f"[Main] Un signal reçu de process {process_id} !")
        except:
            print("[Main] Signal reçu, mais queue vide ?")
        event.clear()
        time.sleep(1)

@ui.page('/')
async def main_page():
    label = ui.label(text='salut')
    set_start_method("spawn", force=True)

    workers = start_workers(event, queue)

    threading.Thread(target=update_ui, args=(label,), daemon=True).start()

    # Supprimer les joins ici, sinon ta page ne s’affiche jamais
    # for w in workers:
    #     w.join()

ui.run()
