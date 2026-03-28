#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для преобразования .bib файла в формат MBB с учетом языка
"""
import os
os.chdir('../outputs/Manuscript (без отметок)')

import re
import sys
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

def detect_language(text):
    """Определяет язык текста (rus/eng)"""
    if not text:
        return "eng"  # по умолчанию английский

    # Проверяем наличие кириллических символов
    cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
    if cyrillic_pattern.search(text):
        return "rus"
    return "eng"

def format_authors(authors, lang="eng"):
    """Форматирование авторов с учетом языка"""
    if not authors:
        return ""

    author_list = authors.split(' and ')

    formatted_authors = []
    for author in author_list[:10]:  # Берем только первых 10
        author = author.strip('{}')

        # Разделяем на фамилию и имена
        parts = author.split(', ')
        if len(parts) >= 2:
            last_name = parts[0].strip()
            first_names = parts[1].strip()

            # Формируем инициалы
            initials = ''.join([name[0].upper() + '.' for name in first_names.split() if name])
            if lang == "rus":
                # Для русских авторов можно оставить как есть или форматировать
                formatted = f"{last_name} {initials}"
            else:
                formatted = f"{last_name} {initials}"
        else:
            formatted = author

        formatted_authors.append(formatted)

    # Добавляем et al./и др.
    if len(author_list) > 10:
        if lang == "rus":
            return ', '.join(formatted_authors) + ' и др.'
        else:
            return ', '.join(formatted_authors) + ' et al.'
    elif len(formatted_authors) == 2:
        if lang == "rus":
            return ' и '.join(formatted_authors)
        else:
            return ' and '.join(formatted_authors)
    else:
        return ', '.join(formatted_authors)

def format_pages(pages, lang="eng"):
    """Форматирование страниц с коротким тире"""
    if not pages:
        return ""

    # Заменяем тире на короткое
    pages = pages.replace('--', '–').replace('-', '–')

    # Проверяем, является ли это Article No.
    if 'e' in pages.lower() or 'article' in pages.lower():
        if lang == "rus":
            return f"Статья № {pages}"
        else:
            return f"Article No. {pages}"
    else:
        if lang == "rus":
            return f"С. {pages}"
        else:
            return f"P. {pages}"

def format_volume_issue(volume, number, lang="eng"):
    """Форматирование тома и номера"""
    parts = []
    if volume:
        if lang == "rus":
            parts.append(f"Т. {volume}")
        else:
            parts.append(f"V. {volume}")
    if number:
        if lang == "rus":
            parts.append(f"№ {number}")
        else:
            parts.append(f"No. {number}")
    return ". ".join(parts)

def format_doi(doi):
    """Форматирование DOI"""
    if not doi:
        return ""

    doi = doi.strip()
    # Убираем префиксы
    if doi.startswith('doi:'):
        doi = doi[4:].strip()
    elif doi.startswith('http://doi.org/'):
        doi = doi[15:]
    elif doi.startswith('https://doi.org/'):
        doi = doi[16:]

    return f"doi: {doi}"

def format_journal(journal, lang="eng"):
    """Форматирование названия журнала"""
    if not journal:
        return ""

    journal = journal.strip('{}')
    # Убираем точку в конце, если есть
    if journal.endswith('.'):
        journal = journal[:-1]
    return journal + "."

def format_title(title, lang="eng"):
    """Форматирование названия статьи"""
    if not title:
        return ""

    title = title.strip('{}')
    # Если название длинное, можно обернуть в кавычки
    if len(title) > 50:
        return f"«{title}»"
    return title

def format_entry(entry):
    """Форматирование одной записи в стиле MBB"""
    # Определяем язык
    lang = entry.get('language', '').lower()
    if not lang:
        # Автоматическое определение по названию журнала и статьи
        journal_lang = detect_language(entry.get('journal', ''))
        title_lang = detect_language(entry.get('title', ''))
        if journal_lang == "rus" or title_lang == "rus":
            lang = "rus"
        else:
            lang = "eng"
    else:
        lang = "rus" if lang.startswith("rus") else "eng"

    # Форматируем авторов
    authors = format_authors(entry.get('author', ''), lang)

    # Название статьи
    title = format_title(entry.get('title', ''), lang)
    if title:
        title = f" {title}."

    # Журнал
    journal = format_journal(entry.get('journal', ''), lang)

    # Год
    year = entry.get('year', '')
    if year:
        year = f"{year}."

    # Том и номер
    volume = entry.get('volume', '')
    number = entry.get('number', '')
    vol_info = format_volume_issue(volume, number, lang)
    if vol_info:
        vol_info = f"{vol_info}."

    # Страницы/Article No.
    pages = format_pages(entry.get('pages', ''), lang)
    if pages:
        pages = f"{pages}."

    # DOI
    doi = format_doi(entry.get('doi', ''))

    # Собираем всё вместе
    parts = []
    if authors:
        parts.append(authors)
    if title:
        parts.append(title.strip())  # Убираем начальный пробел
    if journal:
        parts.append(journal)
    if year:
        parts.append(year)
    if vol_info:
        parts.append(vol_info)
    if pages:
        parts.append(pages)
    if doi:
        parts.append(doi)

    # Убираем пустые элементы
    parts = [p for p in parts if p]

    # Соединяем части
    result = " ".join(parts)

    # Убираем возможные двойные точки в конце элементов
    result = re.sub(r'\.\.', '.', result)
    result = re.sub(r'\.\s*\.', '. ', result)

    return result

def main():
    # Читаем .bib файл
    try:
        with open('Bibliography.bib', 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser(common_strings=True)
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bib_file, parser=parser)

            # Добавляем поле language
            for entry in bib_database.entries:
                if 'language' not in entry:
                    # Определяем язык по журналу и названию
                    journal_lang = detect_language(entry.get('journal', ''))
                    title_lang = detect_language(entry.get('title', ''))

                    if journal_lang == "russian" or title_lang == "russian":
                        entry['language'] = 'russian'
                    else:
                        entry['language'] = 'english'

    except FileNotFoundError:
        print("Ошибка: Файл Bibliography.bib не найден!")
        return

    # Читаем .tex файл для определения порядка цитирования
    try:
        with open('Manuscript.tex', 'r', encoding='utf-8') as tex_file:
            content = tex_file.read()

        # Ищем все команды \cite
        cites = re.findall(r'\\cite\{([^}]+)\}', content)

        # Собираем ключи в порядке упоминания
        ordered_keys = []
        for cite_group in cites:
            for key in cite_group.split(','):
                key = key.strip()
                if key and key not in ordered_keys:
                    ordered_keys.append(key)
    except FileNotFoundError:
        print("Файл Manuscript.tex не найден. Использую алфавитный порядок.")
        ordered_keys = sorted(bib_database.entries_dict.keys())

    # Создаем форматированный список литературы
    output_lines = []
    for i, key in enumerate(ordered_keys, 1):
        if key in bib_database.entries_dict:
            entry = bib_database.entries_dict[key]
            formatted = format_entry(entry)

            # Убираем точку в начале строки, если есть
            if formatted.startswith('.'):
                formatted = formatted[1:].strip()

            output_lines.append((i, key, formatted))
        else:
            print(f"Предупреждение: ключ '{key}' не найден в .bib файле")
            output_lines.append((i, key, f"[СТАТЬЯ НЕ НАЙДЕНА: {key}]"))

    # Сохраняем в файл
    with open('bibliography_mbb.tex', 'w', encoding='utf-8') as out_file:
        out_file.write("% Список литературы в формате MBB\n")
        out_file.write("% Сгенерировано автоматически\n\n")
        out_file.write("\\begin{thebibliography}{99}\n\n")

        for i, key, formatted in output_lines:
            out_file.write(f"\\bibitem{{{key}}}\n")
            out_file.write(f"\\bibentry{{{formatted}}}\n\n")

        out_file.write("\\end{thebibliography}\n")

    print(f"Создан файл bibliography_mbb.tex с {len(output_lines)} записями")

    # Создаем файл для ручной проверки
    with open('bibliography_check.txt', 'w', encoding='utf-8') as check_file:
        check_file.write("ПРОВЕРКА ФОРМАТИРОВАНИЯ БИБЛИОГРАФИИ\n")
        check_file.write("=" * 60 + "\n\n")

        for i, key, formatted in output_lines:
            check_file.write(f"{i}. {formatted}\n")
            check_file.write("-" * 60 + "\n")

    print("Создан файл bibliography_check.txt для проверки форматирования")

if __name__ == '__main__':
    main()