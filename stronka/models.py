from django.db import models

class Baza(models.Model):
    numer_obrazka = models.IntegerField()

    @staticmethod
    def getInstance():
        if Baza.objects.count() > 0:
            return Baza.objects.all()[0]
        else:
            baza = Baza.objects.create(numer_obrazka=0)
            baza.save()
            return baza

    @staticmethod
    def nextNumerObrazka():
        baza = Baza.getInstance()
        baza.numer_obrazka += 1
        baza.save()
        return baza.numer_obrazka

    @staticmethod
    def resetNumerObrazka():
        baza = Baza.getInstance()
        baza.numer_obrazka = 0
        baza.save()
