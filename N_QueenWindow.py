from N_QueenNeuralNetwork import N_QueenNeuralNetwork
import tkinter
import time


class N_QueenWindow(tkinter.Tk):
    def __init__(self, size, n):
        super(N_QueenWindow, self).__init__()
        self.size = size
        self.n = n
        self.solve_n_queen(n)
        self.rect_size = (size-20.0) / n
        self.title(u"N-QUEEN")
        self.geometry("%dx%d" % (size, size))
        self.canvas = tkinter.Canvas(self, width=size, height=size)
        self.canvas.place(x=0, y=0)
        self.cx, self.cy = 10, 10
        for x in range(n):
            for y in range(n):
                self.canvas.create_rectangle(
                    self.cx+x*self.rect_size, self.cy+y*self.rect_size,
                    self.cx+(x+1)*self.rect_size, self.cy+(y+1)*self.rect_size,
                    outline='#405060', tag='board')

    def solve_n_queen(self, n):
        nqnn = N_QueenNeuralNetwork(n)
        self.queens_list = []
        while not nqnn.check():
            nqnn.train()
            queens = nqnn.get_queen_points()
            if not queens in self.queens_list:
                self.queens_list.append(queens)

    def update_canvas(self):
        canvas, cx, cy, rect_size, queens_i, queens_list = self.canvas, self.cx, self.cy, self.rect_size, self.queens_i, self.queens_list
        queens = queens_list[queens_i]
        canvas.delete('queen', 'prog')
        for x, y in queens:
            if self.queen_check(x, y, queens):
                canvas.create_oval(cx+x*rect_size+2, cy+y*rect_size+2,
                                   cx+(x+1)*rect_size-2, cy+(y+1)*rect_size-2,
                                   fill='#50B050', outline='#40A040', tag='queen')
            else:
                canvas.create_oval(cx+x*rect_size+2, cy+y*rect_size+2,
                                   cx+(x+1)*rect_size-2, cy+(y+1)*rect_size-2,
                                   fill='#E06060', outline='#D05050', tag='queen')
        canvas.create_text(self.size-40, self.size-20, tag='prog',
                           text='%d / %d' % (queens_i+1, len(queens_list)))
        if len(queens_list) > queens_i+1:
            sleepTime = 500 - (len(queens_list) - queens_i) * 3
            self.after(sleepTime, self.update_canvas)
        self.queens_i += 1

    def queen_check(self, x, y, queens):
        n = self.n
        for p in range(n):
            if (p != x and (p, y) in queens) or (p != y and (x, p) in queens):
                return False
        for k in range(-min(x, y), min(n-x, n-y)):
            if k != 0:
                if (x+k, y+k) in queens:
                    return False
        for k in range(-min(x, n-y-1), min(n-x, y+1)):
            if k != 0:
                if (x+k, y-k) in queens:
                    return False
        return True

    def mainloop(self):
        self.queens_i = 0
        self.after(500, self.update_canvas)
        super(N_QueenWindow, self).mainloop()
