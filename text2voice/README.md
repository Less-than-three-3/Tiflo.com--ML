# Tiflo.com--ML

<b>Запускаем всю эту красоту</b>

Dockerfile лежит в дерриктории docker, рядом с ним лежит файлик requirements.txt. Стягиваем репу, заходим в эту деррикторию, билдим образ:

<code> docker build -t model_image .</code>

Докер образ с установкой всех зависимостей собирается порядка 20-30 минут и весит около 5-6 гб :)))) \
В некоторые моменты может останавливаться и ничего не писать в консоль. Просто ждём, не паникуем.\
Заааапускаем из образа:

<code>docker run -p 8080:8080 -v <Пусть к файликам из этой репы>:/app -v <Пусть для сохранения аудио файлов на тачке>:/data --name model model_image</code>

Теперь пару слов о том, как этим пользоваться: \
Кидаем запрос на AIService/VoiceTheText в теле указываем текст, который необходимо озвучить в виде жысона, текст на русском, кладём в поле text:

<code>{"text": "Я видел порнофильм, который начинается абсолютно так же"}</code>

В ответ получаем имя аудио файлика, в котором лежит наш озвученный текст. Файлик уже лежит на тачке, по пути указанному выше

<b>Да прибует с нами сила</b>