# pp-2024-test

## Инструкция

* Перед запуском нужно уставить зависимости из `requirements.txt`.
* При первом запуске программа автоматически скачает с гугл диска файлы моделей
в папку `models`.
* Первый аргумент `src` это либо путь к файлу либо url.
* Второй аргумент `target` это путь к папке куда будут записаны результаты,
программа создаст папку если такой не существует.
* Флаг `--first` задает количество удачных кадров до преждевременной остановки
программы. Кадр считается удачным если на нем было распознано хотя бы одно
нужное лицо. При этом, если не задавать этот флаг или вписать число меньшее
единицы, то программа будет работать до окончания потока.
* Флаги `--face`, `--right` и `--left` определяют то, какие классы лиц будут
сохранены в папке `target`. Для тз которое вы описали достаточно `--face`.
* Эта же информация только на корявом английском доступна по команде
`python main.py -h`
