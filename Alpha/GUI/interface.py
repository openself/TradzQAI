from tkinter import *
from ttkthemes import themed_tk as tk
import tkinter.ttk as ttk

from threading import Thread

import pandas as pd
import numpy as np

import train
from environnement import *
from GUI.model_w import *
from GUI.overview_w import *

env = environnement()
style.use('ggplot')

sty = 'arc'


class interface(ttk.Frame):

    def __init__(self, window, **kwargs):
        self.name = "TradzQAI"
        self.version = "Alpha v0.1"

        window.title(self.name + " " + self.version)
        #window.resizable(0, 0)
        ttk.Frame.__init__(self, window, **kwargs)
        self.pack(fill=BOTH)

        self.m = train_thread()
        self.m.daemon=True

        self.build_start_window()

    def manage_orders(self):
        if env.POS_BUY > -1 or env.POS_SELL > -1:
            if "SELL" in env.corder:
                POS = env.POS_BUY
                c = env.sell_price
            elif "BUY" in env.corder:
                POS = env.POS_SELL
                c = env.buy_price
            if len(env.inventory['Price']) > 0:
                new = [str((env.inventory['POS']).iloc[POS]) + " : " + '{:.2f}'.format(env.cd) + " -> " + str(env.corder) + " : " + '{:.2f}'.format(c) + " | Profit : " + '{:.2f}'.format(env.profit)]
                if len(self.o_w.ordr['Orders']) > 39:
                    self.o_w.ordr = (self.o_w.ordr.drop(0)).reset_index(drop=True)
                tmp = pd.DataFrame(new, columns = ['Orders'])
                self.o_w.ordr = self.o_w.ordr.append(tmp, ignore_index=True)

    def build_start_window(self):
        self.start = ttk.Frame(self)
        self.start.pack()

        self.smod = ttk.LabelFrame(self.start, text="Mode")
        self.smod.pack(side=RIGHT, fill=BOTH)

        self.train = ttk.Button(self.smod, text="Train", command=self.build_train_window)
        self.train.pack(side=TOP, fill=BOTH, expand="yes")
        self.eval = ttk.Button(self.smod, text="Eval", command=self.build_eval_window)
        self.eval.pack(side=TOP, fill=BOTH, expand="yes")

        self.es = ttk.LabelFrame(self.start, text="Environnement settings")

        # Model choice

        self.fesm = ttk.Frame(self.es)
        self.esm = ttk.Label(self.fesm, text="Model : ")
        self.mv = StringVar()
        self.mv.set(str(env.stock_name))
        self.eesm = ttk.Entry(self.fesm, textvariable=self.mv)

        # Max order choice

        self.fesmo = ttk.Frame(self.es)
        self.esmo = ttk.Label(self.fesmo, text="Max order : ")

        self.mov = StringVar()
        self.mov.set(str(env.max_order))

        self.smo = Spinbox(self.fesmo, from_=0, to=100, textvariable=self.mov)

        # Contract price choice

        self.fescp = ttk.Frame(self.es)
        self.escp = ttk.Label(self.fescp, text="Contract price : ")

        self.cpv = StringVar()
        self.cpv.set(str(env.contract_price))

        self.scp = Spinbox(self.fescp, from_=1, to=10000, textvariable=self.cpv)

        # Spread choice

        self.fesp = ttk.Frame(self.es)
        self.esp = ttk.Label(self.fesp, text="Spread : ")

        self.sv = StringVar()
        self.sv.set(str(env.spread))

        self.ss = Spinbox(self.fesp, from_=1, to=10, textvariable=self.sv)

        # Packing

        self.es.pack(side=LEFT)

        self.fesm.pack(anchor='e')
        self.esm.pack(side=LEFT)
        self.eesm.pack(side=RIGHT)

        self.fesmo.pack(anchor='e')
        self.esmo.pack(side=LEFT)
        self.smo.pack(side=RIGHT)

        self.fescp.pack(anchor='e')
        self.escp.pack(side=LEFT)
        self.scp.pack(side=RIGHT)

        self.fesp.pack(anchor='e')
        self.esp.pack(side=LEFT)
        self.ss.pack(side=RIGHT)

    def build_primary(self):
        # Getting env settings
        env.contract_price = int(self.scp.get())
        env.stock_name = str(self.eesm.get())
        env.spread = int(self.ss.get())
        env.max_order = int(self.smo.get())

        self.start.destroy()

        window.geometry("1024x720")

        self.note = ttk.Notebook(window)

        self.base = ttk.Frame(self.note)
        self.model = ttk.Frame(self.note)
        self.historic = ttk.Frame(self.note)

        self.quit = ttk.Button(self, text="Quit", command=quit)
        self.quit.pack(side=RIGHT, fill=BOTH)
        self.run_ai = ttk.Button(self, text="Run", command=self.go_run)
        self.run_ai.pack(side=LEFT, fill=BOTH)
        self.parsing = ttk.Button(self, text="Pause", command=self.m.pause)
        self.parsing.pack(side=LEFT, fill=BOTH)
        self.but = ttk.Button(self, text="Resume", command=self.m.resume)
        self.but.pack(fill=BOTH, side=LEFT)

        self.note.add(self.base, text="Overview")
        self.note.add(self.model, text="Model")
        self.note.add(self.historic, text="Historic")

        self.note.pack(fill=BOTH, expand="yes")

        self.m_w = model_window(self.model, env)
        self.o_w = overview_window(self.base, env)

    def build_train_window(self):
        env.mode = "train"
        self.build_primary()

    def build_eval_window(self):
        env.mode = "eval"
        self.build_primary()

    def go_run(self):
        self.m.start()
        self.run_ai.destroy()

    def update(self):
        # Process order
        self.manage_orders()

        # Update model graph
        self.m_w.update_graph(env)

        # Update Overview
        self.o_w.update_overview(env)

class train_thread(Thread):

    def __init__(self):
        Thread.__init__(self)

    def pause(self):
        self.t.b = 1

    def resume(self):
        self.t.b = 0

    def run(self):
        self.t = train.train(env, interfaces)
        self.t.training(env)


def launch_interface():
    interfaces.mainloop()

window = tk.ThemedTk()
window.set_theme(sty)
interfaces = interface(window)
