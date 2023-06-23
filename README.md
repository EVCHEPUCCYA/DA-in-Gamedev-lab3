# DA-in-Gamedev-lab3
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Лутков Евгений Александрович
- РИ210933
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Познакомиться с созданием системы машинного обучения и интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.

При выполнении задания пользовался методическими указаниями по лабораторной работе.

Добавил MLAgent в Unity
![2023-06-01_19-25-28](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/102b64b4-72f0-4d27-8206-d7673c1e01d5)
Также, с помощью Anaconda активировал ML агента и скачал mlagents 0.28.0 и torch 1.7.1
![2023-06-01_17-04-23](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/570f3edb-d075-46ef-ad97-cd4b64466d4a)

Создал на сцене плоскость, куб и сферу, добавляю скрипт.
![2023-06-01_18-01-51](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/eefcbd5b-18db-4778-b756-d16de7e68289)

Скрипт:
```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
Добавил Decision Requester, Behavior Parameters к сфере.
![2023-06-01_19-37-36](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/ef0b19e8-1191-4e80-b112-3604e3091b40)

Добавил параметры нейросети в корень проекта:
```
behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```
Запуски на 1, 3, 9 и 27 прошли без ошибок:
![2023-06-01_19-37-36](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/c18aa01e-425c-4e8c-9a7a-be198aa4e296)
![2023-06-01_19-39-21](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/54c2ff9f-f973-4f73-b77a-82c6df3cb94c)
![2023-06-01_19-39-54](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/12f12e11-a391-450d-916b-075164752704)
![2023-06-01_19-39-54](https://github.com/EVCHEPUCCYA/DA-in-Gamedev-lab3/assets/113372135/2ed85b5e-b508-4a7e-89bb-35489bb3d815)


## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

DecisionRequest - это компонент, который определяет частоту, с которой агент запрашивает решение.
Behaviors Parameters - это компонент для настройки поведения экземпляра агента и его свойств - имя, тип поведения, версию нейросети и т.д.
```
behaviors:
  RollerBall:
    trainertype: ppo # Используется алгоритм PPO для обучения - с подкреплением
    hyperparameters:
      batchsize: 10      # Размер числа опытов
      buffersize: 100    # Размер необходимого опыта для обновления модели поведения
    learning_rate: 3.0e-4    # Начальная скорость обучения для градиентного спуска. С
      beta: 5.0e-4 # Сила регуляризации энтропии, которая делает политику "более случайной".
      epsilon: 0.2 # Влияет на то, насколько быстро политика может развиваться во время обучения.
      lambd: 0.99 # Насколько агент полагается на свою текущую оценку значений при расчете предсказаний. Высокие значения соответствуют тому, что агент больше полагается на фактические вознаграждения, полученные в окружающей среде
      num_epoch: 3  # Количество проходов, которые необходимо выполнить через буфер опыта при выполнении оптимизации градиентного спуска.
      learning_rate_schedule: linear  # Определяет, как скорость обучения меняется с течением времени.
    network_settings:    # Настройки нейронной сети.
      normalize: false   # Применяется ли нормализация к входным данным векторного наблюдения. 
      hidden_units: 128     #К оличество нейронов в скрытых слоях нейронной сети.
      num_layers: 2      # Количество скрытых слоев в нейронной сети. Соответствует количеству скрытых слоев, присутствующих после входящих данных
    reward_signals:    # Настройки для внешних и внутренних сигналов вознаграждения
      extrinsic: # Внешние награды.
        gamma: 0.99   # Коэффициент дисконтирования для будущих вознаграждений
        strength: 1.0   # Значение, на которое можно умножить вознаграждение, получаемое от окружающей среды.
    max_steps: 500000    # Количество повторов.
    time_horizon: 64     # Сколько шагов опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта.
    summary_freq: 10000    # Частота сохранения сводки для записи в файлы во время обучения.
```
## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

## Выводы
В этой лабораторной работе я научился интегрировать нейронную сеть в проект Unity. Возникли проблемы с движением объектов - не срабатывала эврестическая модель поведения, и сфера не двигалась, из-за чего два раза пришлось начинать делать проект с нуля.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
