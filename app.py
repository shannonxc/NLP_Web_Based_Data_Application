from flask import Flask, render_template, request, redirect
from gensim.models.fasttext import FastText
import pickle
import os
from bs4 import BeautifulSoup
from utils import gen_docVecs, get_filenames_by_folder

app = Flask(__name__)
app.secret_key = os.urandom(16) 


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/accounting')
def accounting():
    filenames = get_filenames_by_folder('Accounting_Finance')
    return render_template('accounting.html',  filenames=filenames)


@app.route('/engineering')
def entertainment():
    filenames = get_filenames_by_folder('Engineering')
    return render_template('engineering.html',  filenames=filenames)


@app.route('/healthcare')
def healthcare():
    filenames = get_filenames_by_folder('Healthcare_Nursing')
    return render_template('healthcare.html',  filenames=filenames)


@app.route('/sales')
def sales():
    filenames = get_filenames_by_folder('Sales')
    return render_template('sales.html',  filenames=filenames)


@app.route('/<folder>/<filename>')
def article(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':

        # Read the content
        f_title = request.form['title']
        f_content = request.form['description']
        f_company = request.form['company']
        f_salary = request.form['salary']

        # Classify the content
        if request.form['button'] == 'Classify':

            # Tokenize the content of the .txt file so as to input to the saved model
            # Here, as an example,  we just do a very simple tokenization
            title_content = f_title + ' ' + f_content
            tokenized_data = title_content.split(' ')

            # Load the FastText model
            jobFT = FastText.load("jobFT.model")
            jobFT_wv = jobFT.wv

            # Generate vector representation of the tokenized data
            jobFT_dvs = gen_docVecs(jobFT_wv, [tokenized_data])

            # Load the LR model
            pkl_filename = "jobFT_LR.pkl"
            with open(pkl_filename, 'rb') as file:
                model = pickle.load(file)

            # Predict the label of tokenized_data
            y_pred = model.predict(jobFT_dvs)
            y_pred = y_pred[0]

            return render_template('admin.html', prediction=y_pred, title=f_title, description=f_content, company=f_company, salary=f_salary)

        elif request.form['button'] == 'Save':
            # First check if the recommended category is empty
            cat_recommend = request.form['category']
            if cat_recommend == '':
                return render_template('admin.html', prediction=cat_recommend,
                                       title=f_title, description=f_content, company=f_company, salary=f_salary,
                                       category_flag='Recommended category must not be empty.  '
                                                     'Example category: Accounting_Finance, '
                                                     'Engineering, Healthcare_Nursing, Sales.')

            elif cat_recommend not in ['Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales']:
                return render_template('admin.html', prediction=cat_recommend,
                                       title=f_title, description=f_content, company=f_company, salary = f_salary,
                                       category_flag='Recommended category must belong to: Accounting_Finance, '
                                                     'Engineering, Healthcare_Nursing, Sales. ')
            else:
                # First read the html template
                soup = BeautifulSoup(open('templates/article_template.html'), 'html.parser')

                # Then adding the title and the content to the template
                # First, add the title
                div_page_title = soup.find('div', {'class': 'title'})
                title = soup.new_tag('h1', id='data-title')
                title.append(f_title)
                div_page_title.append(title)

                # Second, add the company
                div_page_company = soup.find('div', {'class': 'data-company'})
                company = soup.new_tag('p')
                company.append("Company Name: " + f_company)
                div_page_company.append(company)

                # third, add the content
                div_page_content = soup.find('div', {'class': 'data-article'})
                content = soup.new_tag('p')
                content.append("Job Description: " + f_content)
                div_page_content.append(content)

                # fourth, add the salary
                div_page_salary = soup.find('div', {'class': 'data-salary'})
                salary = soup.new_tag('p')
                salary.append("Salary: " + f_salary)
                div_page_salary.append(salary)

                # Finally write to a new html file
                filename_list = f_title.split()
                filename = '_'.join(filename_list)
                filename = cat_recommend + '/' + filename + ".html"
                with open("templates/" + filename, "w", encoding='utf-8') as file:
                    print(filename)
                    file.write(str(soup))

                # Redirect to the newly-generated news article
                return redirect('/' + filename.replace('.html', ''))

    else:
        return render_template('admin.html')