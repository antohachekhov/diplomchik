class Line:
    """
        Объект класса - прямая линия,
        описываемая каноническим уравнением прямой:
        ax + by + c = 0

        Атрибуты
        ----------
        a : float, по умолчанию = 1
            Коэффициент при x в каноническом уравнении прямой

        b : float, по умолчанию = 1
            Коэффициент при y в каноническом уравнении прямой

        с : float, по умолчанию = 0
            Свободный коэффициент в каноническом уравнении прямой
    """

    def __init__(self, a=1., b=1., c=0.):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """
        Возвращает значение y прямой при заданной координате x

        :param x: float
            Значение x, в котором нужно найти точку прямой
        """
        return (self.a / self.b) * x + (self.c / self.b)

    def y(self, x):
        """
        Возвращает значение y прямой при заданной координате x

        :param x: float
            Значение x, в котором нужно найти точку прямой
        """
        return (self.a / self.b) * x + (self.c / self.b)

    def generalEquation(self, points):
        """
        Возвращает значение канонического уравнения прямой для заданной точки (x, y)

        :param points: ndarray
            Точка или набор точек (x, y), для которых нужно найти значение канонического уравнения
        """
        if points.ndim == 1:
            # points - одна точка
            if points.shape[0] != 2:
                # points не является точкой в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[0] - self.b * points[1] + self.c
        elif points.ndim == 2:
            # points - набор точек
            if points.shape[1] != 2:
                # points не являются точками в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, 0] - self.b * points[:, 1] + self.c
        elif points.ndim == 3:
            # points - набор массивов с одной точкой
            if points.shape[2] != 2:
                # points не являются точками в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, :, 0] - self.b * points[:, :, 1] + self.c
        else:
            raise ValueError('Points have incorrect dimension')
